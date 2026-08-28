from __future__ import annotations

import os
import re
import json
import urllib.error
import urllib.request
from datetime import date
from decimal import Decimal
from typing import Any

import google.generativeai as genai
from psycopg import sql as pg_sql
from psycopg import Error as PostgresError
from psycopg.rows import dict_row

from db import connect, get_database_url, get_default_schema, get_schemas
from lightweight_rag import format_rejected_context, format_retrieved_context

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com/api")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-coder:480b-cloud")
# In-cluster LiteLLM proxy (OpenAI-compatible), same pattern as the sibling "datachat" app.
DEFAULT_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://litellm.litellm.svc.cluster.local:4000/v1")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "qwen3.6-35b-a3b-coder")
MAX_QUERY_ROWS = int(os.getenv("MAX_QUERY_ROWS", "5000"))

SCHEMA_PROMPT_TEMPLATE = """
You are writing PostgreSQL SELECT queries for a basketball analytics database.

Today's date is {today}.

Schemas: {schemas}

Domain notes:
- CRITICAL: resolve relative season words against today's date above, and remember a
  season is named for the two calendar years it spans (Aug of the first year to Jul of
  the second). "Last season" / "letzte Saison" means the most recently COMPLETED season,
  which is NOT simply "today's year minus one": on 2026-08-27 the 2025-2026 season ended
  in June 2026 and the 2026-2027 season has barely started, so "last season" is
  2025-2026 (date >= '2025-08-01' AND date < '2026-08-01'), not 2024-2025. Likewise
  "this season" means the season currently in progress, and "the last N seasons" counts
  back from the most recently completed one. If a question names a season explicitly
  ("2024-25", "season 2024"), use that instead and ignore today's date.
- Table prefixes indicate competitions: b_el = EuroLeague, b_ec = EuroCup, b_cl = Champions League, b_bbl = Basketball Bundesliga.
- boxscore tables are usually best for player/team game totals and rankings.
- CRITICAL: in b_el_boxscore, b_ec_boxscore, b_cl_boxscore, AND the corresponding *_playbyplay tables (b_el_playbyplay, b_ec_playbyplay, b_cl_playbyplay), home_team_total_point and away_team_total_point are repeated on EVERY row of that game - one row per player in boxscore (~23-24 rows per team per game), one row per EVENT in playbyplay (can be 500+ rows per game) - all carrying the identical team-total value. Summing these columns per row overcounts a team's points by roughly the roster size (boxscore) or the event count (playbyplay, far worse). This is independent of and more fundamental than any team-name-rename issue below: before summing a team's points across games, ALWAYS collapse to one row per team-game first (e.g. GROUP BY team, link and take MAX(home_team_total_point)/MAX(away_team_total_point), or an equivalent dedup), THEN sum across games. Never write "SUM(home_team_total_point) ... UNION ALL ..." directly against the raw per-player or per-event rows.
- CRITICAL: this same duplication also breaks COUNT(*) for "games played" or "W-L record" questions on these same tables - COUNT(*) counts RAW ROWS (one per player in boxscore, one per event in playbyplay), not games. Verified against live data: a head-to-head "games played" question between two EuroLeague teams using plain COUNT(*) returned 48 (24 players x 2 real games), when the real answer was 2 games - a 24x inflation, and the resulting "48-0" record was equally wrong (real record: 2-0). To count actual games, either COUNT(DISTINCT link) directly, or first collapse to one row per game (e.g. a subquery/CTE with SELECT DISTINCT link, home_team, away_team, home_team_total_point, away_team_total_point) before counting or computing a win/loss record from it.
- CRITICAL: when computing one specific team's win/loss outcome per game (not just a raw team-total aggregation), you MUST check BOTH the case where that team was home AND the case where it was away - a CASE/WHEN that only tests "team ILIKE home_team AND home_pts > away_pts" silently drops every game where that team played away, undercounting both wins and losses. Verified against live data (head-to-head, 2 games): checking only the home-team win condition reported Real Madrid 1-0 against Barcelona; the real record, checking both sides symmetrically, is 2-0. For a small filtered set, one CASE covering "(team ILIKE home_team AND home_pts > away_pts) OR (team ILIKE away_team AND away_pts > home_pts)" works. For a FULL-SEASON record (many opponents, using the UNION ALL team-reorientation pattern from the team-totals note above), do NOT keep separate home_pts/away_pts columns and compare them directly after the UNION - "home_pts > away_pts" silently means something different depending on which branch a row came from, and checking it uniformly gives a WRONG record even though games_played comes out correct. Verified against live data: this exact mistake reported Real Madrid 35-9 for the full 2025-26 season; the real record, from correctly reorienting each branch to team_pts/opp_pts (SELECT home_team AS team, home_team_total_point AS team_pts, away_team_total_point AS opp_pts ... UNION ALL SELECT away_team AS team, away_team_total_point AS team_pts, home_team_total_point AS opp_pts ...), is 28-16 - then win/loss is simply team_pts > opp_pts / team_pts < opp_pts, correct for every row regardless of which branch it came from.
- playbyplay tables are usually best for event sequences, shot/action timing, and possession-level questions.
- player_info tables are usually best for player lookup and roster attributes.
- wurfposition means shot position/location.
- b_bbl_boxscore uses date_final, home_team_final, away_team_final instead of date, home_team, away_team used by b_el_boxscore, b_ec_boxscore, b_cl_boxscore.

Season and date filtering (read this before writing any query that mentions a season or year):
- bronze.* boxscore/playbyplay tables have NO season column, only a per-game date (date_final for BBL). If a question names a season or year and the query has no date/season filter, the query is wrong, not just imprecise - it will silently mix multiple seasons together.
- A competition season "YYYY-(YYYY+1)" runs from around August of YYYY to around July of YYYY+1, crossing the calendar-year boundary. "Season 2025-2026" and "season 2025" both mean this range - NEVER filter by calendar year (date BETWEEN '2025-01-01' AND '2025-12-31'), that silently cuts the season in half.
- For a bronze-only query scoped to season YYYY-(YYYY+1), filter with: date >= 'YYYY-08-01' AND date < '{{YYYY+1}}-08-01' (use date_final for BBL).
- gold.g_el_players, g_ec_players, g_cl_players, and g_bbl_players already have a saison column formatted like '2025-2026' - prefer filtering there with saison = '2025-2026' over date math whenever the requested stat exists in a gold table.
- CRITICAL: gold.g_player_profile_* tables (pts, ast, reb, etc.) store PER-GAME AVERAGES, not season totals, and carry NO minimum-games qualification - a player with gp=1 who scored 21 in that one game outranks players who averaged 19 across 37 games. A "top scorer"/"best"/"leading" question ranked by an unqualified per-game average is misleading, not just imprecise (verified against live data: EuroLeague 2025-26's #1 by raw pts is a 1-game player, not the real 39-game, 740-total-point scoring leader). Prefer adding a minimum games-played threshold ON THIS SAME TABLE (e.g. gp >= 20, roughly half a season) over switching to a different table - g_player_profile_* already has position/team/nationality alongside the stat in one row, so filtering it directly avoids needing any join at all. Only fall back to SUM(pts) from a bronze boxscore table if the question specifically needs a true season total rather than a qualified average - and if you do, see the player-name-format note below before joining boxscore to any other table for extra attributes. TWO EXCEPTIONS to "prefer this table". (a) Because the columns are ALREADY per-game, NEVER divide them by gp - that yields a per-game-per-game figure that systematically favours players with few games; verified against live data, "trb / gp" picked the wrong best rebounder for 8 of 18 BBL teams, promoting a 16-game player who rebounds worse per game over a 21-game player who rebounds better. Rank on the column as it stands. (b) These tables are INCOMPLETE, so any question about coverage - "per team", "every player", "how many players" - must be answered from the bronze boxscore instead; verified against live data, g_player_profile_bbl holds only 265 players for 2025-2026 and is missing real rotation players entirely (Shittu, Klassen, both among the league's top rebounders), so a per-team ranking built on it silently omits several teams' actual leader.
- CRITICAL: player_name format is NOT consistent across tables, even within one league - verified against live data: bronze.b_el_boxscore uses "LAST, FIRST" (comma, e.g. "VEZENKOV, SASHA") while bronze.b_el_player_info uses "FIRST LAST" (no comma, e.g. "SASHA VEZENKOV"); other leagues' player_info/boxscore pairs may differ again. A join on player_name across two different tables can silently return zero rows ("no data found") instead of erroring, which looks like a missing-data answer but is actually a wrong query. Before ever joining bronze boxscore to player_info (or any other table) on player_name, sample a few player_name values from both tables to confirm the format actually matches, or normalize one side first. Prefer a gold table that already has the attributes you need in one row (see the gp-qualification note above) over a fragile cross-table name join.
- CRITICAL: a "best ratio" question (assist-to-turnover, steal-to-foul, or any ratio of two counting stats) needs a minimum-VOLUME floor on top of any minimum-games floor, or a near-zero-usage player wins by having proportionally tiny numbers on both sides of the ratio. gp >= 20 alone does not fix this - verified against live data: EuroLeague 2025-26's #1 by raw ast/tov ratio (gp >= 20) is a player averaging 0.5 ast and 0.09 tov per game (barely touches the ball), ratio 5.56; adding a realistic volume floor on the numerator stat (e.g. ast >= 3 per game, a normal analytical threshold for a real playmaker) gives a completely different and correct answer: 5.13 ast / 1.26 tov, ratio 4.07. Always add a volume floor on whichever stat represents meaningful usage, not just a games-played floor, when ranking by a ratio.
- CRITICAL: never guess a position value (e.g. 'C', 'PG') - always use the exact stored string, which differs by league and is NOT always an abbreviation. Verified against live data, position column in *_player_info tables: EL has exactly ('Guard', 'Forward', 'Center') - full words, no abbreviations exist, so position = 'C' matches zero rows and silently returns NULL instead of erroring. EC has the same three full words plus a blank string for some rows. CL mixes two granularities: ('Guard', 'Forward', 'Center') AND ('Point Guard', 'Shooting Guard', 'Small Forward', 'Power Forward') in the same column - a query for "guards" should typically match both 'Guard' and 'Point Guard'/'Shooting Guard' (e.g. position ILIKE '%guard%'), not just one form. BBL is the messiest: both full-name and abbreviation forms coexist, sometimes as one string like 'Power Forward (PF)' and sometimes as bare 'PF' or 'Center (C)' vs 'C' - match BBL positions with ILIKE '%pattern%' (e.g. '%center%' or '%(C)%'/'%\\bC\\b%'), never exact equality. Because these conventions differ so much, positions are also not directly comparable across leagues in the same query.
- CRITICAL: this applies to ANY flag/success/boolean-like text column, not just position - never guess its encoding ('1'/'0', 'Y'/'N', 'true'/'false' are all wrong guesses unless verified). Verified against live data: bronze.*_wurfposition.success uses lowercase 'yes'/'no' (success = '1' matches zero rows, silently giving a 0.0% shooting percentage instead of erroring or asking); bronze.b_bbl_boxscore.s_five uses 'S5' for a starter and an empty string for a non-starter (not 'Y'/'N'). When a question needs a flag-style column whose exact stored values you have not already seen in this conversation, prefer a query shape that does not require guessing (e.g. GROUP BY the column itself to see its real values) over assuming a common convention.
- CRITICAL: *_player_info tables (position, nationality, height, birthday) have NO season or date column - they are a rolling multi-season/career player registry, not one season's roster, and the same player can appear multiple times across different crawled seasons or team changes. A question that names a season or says "this season" but is answered purely from *_player_info silently mixes every season the crawler has ever seen together instead of scoping to one - verified against live data: counting Champions League guards purely from b_cl_player_info gives 537 (all crawled seasons combined), but the real 2025-26-season count, found by requiring the player to also appear in that season's date-filtered boxscore, is 212 - a ~2.5x inflation. Whenever a question about position/nationality/height/etc. also names a season, first find that season's actual player set from a season-dated source (bronze boxscore filtered by date, or a gold table with saison) and only THEN join to player_info for the attribute - never query player_info alone when the question specifies a season. (player_name format usually matches between a league's own boxscore and player_info tables, but confirm before joining per the player-name-format note above - it is not guaranteed.)
- CRITICAL: the nationality column is encoded DIFFERENTLY IN EVERY LEAGUE, so a country filter that works for one league silently returns zero rows for another - verified against live data, WHERE nationality ILIKE '%Germany%' on b_bbl_player_info matches 0 of 859 rows because BBL stores it in German. The three encodings: b_bbl_player_info uses German country names with a 2-letter ISO code in brackets ('Deutschland (DE)', 'Vereinigte Staaten von Amerika (US)', 'Kanada (CA)') and INCONSISTENTLY - 260 rows say 'Deutschland (DE)' but another 70 say just 'Deutschland'. b_el_player_info and b_ec_player_info use English full names ('United States of America', 'Germany', and note EC writes 'Turkiye', not 'Turkey'). b_cl_player_info uses 3-letter uppercase codes ('GER', 'USA', 'FRA'). Two consequences. (a) NEVER compare nationality with '=' - always ILIKE '%...%', because equality misses both the bracket-less BBL variant and every dual national. (b) Dual nationality is stored inside the single cell, separated by a NEWLINE in BBL (86 of 859 rows, e.g. 'Deutschland (DE)' + newline + 'Vereinigte Staaten von Amerika (US)') and by a COMMA in CL (203 of 1407 rows, e.g. 'BIH, SRB'); a substring ILIKE handles both, an equality or a GROUP BY on the raw value treats each combination as its own country. There is no normalised country column anywhere, so a question spanning several leagues cannot be answered with one literal - filter each league with its own spelling.
- CRITICAL: for BBL specifically, b_bbl_boxscore.player_name ("Mikesell R. (SF)" - abbreviated first name, trailing "(POS)" suffix, mixed case) does NOT match b_bbl_player_info.player_name ("RYAN MIKESELL" - full name, all caps, no suffix) at all - joining on player_name directly returns zero rows every time, verified against live data (a BBL center count came back 0 instead of the real 56 this way). The bridge column is b_bbl_player_info.player_name_short ("MIKESELL R." - all caps, no suffix): strip the "(POS)" suffix from boxscore's player_name and uppercase it before joining, e.g. UPPER(regexp_replace(boxscore.player_name, '\\s*\\(.*\\)\\s*$', '')) = player_info.player_name_short. Also note b_bbl_player_info can carry multiple, sometimes conflicting, position strings for the same player across different crawled records (e.g. one row says 'Small Forward (SF)', another says 'SF', another says 'Shooting Guard (SG)' for the same person) - use ILIKE matching (per the position-value note above) rather than expecting one canonical value per player.
- CRITICAL: get the sort direction right for "youngest"/"oldest" and similar chronological superlatives - a birthday column sorted ASC gives the EARLIEST date, which is the OLDEST person, not the youngest. "Youngest" needs ORDER BY birthday DESC (most recent birthdate first); "oldest" needs ORDER BY birthday ASC. Verified against live data: ORDER BY birthday ASC LIMIT 1 for "youngest player in the EuroLeague" wrongly returned a player born in 1986 (actually the oldest in the dataset); the real youngest, found with DESC, was born in 2009 - a completely different, wrong person, not just an imprecise one. Double-check this inversion risk on any "most recent"/"earliest"/"newest"/"latest" question involving a date column too, not just birthday.
- CRITICAL: of_per_eigenes/of_per_home/of_per_away/of_per and their def_per_* counterparts (gold.g_*_teams, silver.*_teams_prospiel) are OFFENSIVE and DEFENSIVE REBOUND percentage, not "offense/defense strength" - the "of"/"def" prefix reads like an overall rating but is not one. Verified against live data: of_per_eigenes + def_per_gegner = 1.000 for every team, every row, which only holds for a rebounding-percentage pair (a team's offensive board rate and its opponent's defensive board rate on the same missed shots are complementary) - no notion of scoring or shooting efficiency produces that identity. A question about a team's "Angriffsstärke"/"offensive strength"/"attacking rating" means pts_per_poss_eigenes (or pts_per_poss_eigenes minus pts_per_poss_gegner for a net rating), never of_per_eigenes.
- CRITICAL: net_rating_off_court (gold.g_bbl_players, and the equivalent in other leagues' player tables) is the TEAM's net rating during the minutes this specific player was NOT on the floor - an on/off-style stat, not the player's own offensive rating and not "rating while playing offense". Verified against live data: values cluster by TEAM, not by individual talent - every FC Bayern München player's net_rating_off_court is positive (12 to 22), every Basketball Löwen Braunschweig player's is negative (-15 to -6), which is the signature of a stat that reflects team strength-without-this-player rather than the player's own production. A question asking for a player's own offensive quality means offensive_rating, not net_rating_off_court - answering with net_rating_off_court silently answers a different, backwards question (a HIGH value here can mean the team does fine without that player, closer to "replaceable" than "outstanding"). There is no net_rating_on_court counterpart column - an on/off DIFFERENCE for a player is not directly queryable and would need deriving from gold.g_*_teams' overall net rating minus this column.
- CRITICAL: ef (bronze.*_boxscore) is the classic basketball "Efficiency" (EFF) rating - (PTS+REB+AST+STL+BLK) minus (missed FG + missed FT + TOV) - an unbounded counting number (can run 40+), NOT effective field goal percentage (eFG%, a 0-100% shooting-efficiency stat computed from FGM/three-pointers/FGA). Verified against live data: a 52-point game (18/25 FG, 7/7 FT, 1 reb, 3 ast, 2 stl, 0 blk, 3 tov) has ef = 48, matching (52+1+3+2+0) - ((25-18)+(7-7)+3) = 58-10 = 48 exactly. A question asking for "eFG%" or "effective field goal percentage" needs FGM/FGA/three_p computed directly (see the eFG% pattern used elsewhere in this file's examples), not the ef column - they are unrelated statistics that happen to share a similar-looking abbreviation.
- The tm-prefixed columns in silver.*_player_stats_prospiel (tmpts, tmast, tmofreb, tmdefreb, tmstl, tmto) are the TEAM's totals accumulated only during this player's on-court minutes for that game - not this player's own individual stat (that is the bare pts/ast/etc. in bronze boxscore tables) and not the team's full-game total. Verified against live data: for a game where the team's real final total was 108 points and 15 assists, one rotation player's tmpts was 95 and tmast was 14 - both slightly under the full-game figures, consistent with "team production while this player was on the floor" for most but not all of the game. Do not sum tmpts across a player's games to get "team points" (it double counts every possession the team's other rotation players were also on court for) and do not treat it as the player's own scoring.
- cnt_angriffe / angriffe_off / angriffe_def (silver.*_player_stats_prospiel, gold.g_bbl_players) use "Angriffe", German for "possessions" in basketball usage here, not "attacks" in any violent or foul-related sense - this is the possession count used to build the pts_per_poss-style rate stats, at the same on-court-subset grain as the tm-prefixed columns above.
- When grouping player stats by season, GROUP BY player_name alone, not (player_name, team). Several teams have mid-season sponsor renames (the same player then has two team-name rows in one season), which silently splits and undercounts a renamed team's players if team is in the GROUP BY.
- The same sponsor-rename problem also breaks TEAM-level aggregation (SUM/COUNT grouped by team, home_team, or away_team): a "which team scored the most" or "top N teams" query silently undercounts and misranks a renamed team unless the name variants are normalized to one canonical name first with a CASE WHEN before grouping. Known pairs (old name -> canonical name, verified against bronze, ALL CAPS): EC 'HAPOEL BANK YAHAV JERUSALEM' -> 'HAPOEL MIDTOWN JERUSALEM'; EC 'PANIONIOS ATHENS' -> 'PANIONIOS COSMORAMA TRAVEL ATHENS'; EL 'PANATHINAIKOS ATHENS' -> 'PANATHINAIKOS AKTOR ATHENS'; EL 'BASKONIA VITORIA-GASTEIZ' -> 'KOSNER BASKONIA VITORIA-GASTEIZ'; EL 'CRVENA ZVEZDA MERIDIAN BELGRADE' -> 'CRVENA ZVEZDA MERIDIANBET BELGRADE'. CRITICAL: gold tables (e.g. g_el_teams) store these SAME team names in Title Case ('Crvena Zvezda Meridian Belgrade'), not ALL CAPS - a CASE WHEN team ... using the ALL-CAPS literal against a Title Case column matches nothing and silently falls through to ELSE, so the rename normalization silently does not fire at all. Always write the rename CASE WHEN case-insensitively regardless of source table - e.g. CASE UPPER(team) WHEN 'CRVENA ZVEZDA MERIDIAN BELGRADE' THEN 'Crvena Zvezda Meridianbet Belgrade' ELSE team END - verified against live data: the case-sensitive version against gold.g_el_teams silently returned the wrong, unmerged small-sample team as the answer with no error at all.
- CRITICAL: this rename split is NOT limited to bronze - gold.g_el_teams, g_ec_teams, g_bbl_teams, etc. (pre-aggregated per-team-season rate stats like pts_per_poss_eigenes) can ALSO have two separate rows for the same team+saison, one per name variant, and these are RATE stats you cannot simply SUM to combine - you must weight by possessions (anz_ballbstz_eigenes). Verified against live data: gold.g_el_teams has both 'Crvena Zvezda Meridian Belgrade' (70.0 possessions/game, an incomplete partial-season split) and 'Crvena Zvezda Meridianbet Belgrade' (74.974 possessions/game) as separate rows for saison 2025-2026 - picking either row alone for a "best net rating" question gives a wrong team AND a wrong number: the properly weighted combined net rating (0.080) is actually LOWER than Olympiacos Piraeus's 0.108, meaning Olympiacos is the real answer, not Crvena Zvezda's misleadingly high 0.159 from the small-sample split alone. Before ranking teams by any gold.g_*_teams column, check whether any team name appears more than once for the same saison (a quick GROUP BY team, saison HAVING COUNT(*) > 1 first); if so, merge the rate columns with a possession-weighted average (SUM(rate * anz_ballbstz_eigenes) / SUM(anz_ballbstz_eigenes)) before ranking, don't just pick one row.
- BBL bronze/silver team rows include ProA cup opponents that are not real Bundesliga teams - they appear only 1-2 times all season (vs. 33+ for a real BBL team), a known data-quality gap with no clean structural filter (no competition/spieltyp column exists yet). Any BBL team-level aggregation ("which team scored the most", "how many teams", "bottom N teams", averages across teams) MUST exclude these with HAVING COUNT(*) > 2 on the team-game grouping, or it silently counts 6-7 phantom cup opponents as BBL teams (24 raw team names vs. ~17-18 real ones). Verified against live data: this never changes who ranks #1 (phantom teams have tiny totals) but corrupts team counts, bottom-N rankings, and averages.
- CRITICAL: a question asking for "the average/mean/typical" value of a stat across players or teams (NOT asking to rank, list, or find the top/best) must be answered with a single aggregate row computed over ALL qualifying rows - e.g. SELECT AVG(pct) FROM (qualifying subquery), or SUM(made)*100.0/SUM(attempts) for a weighted percentage. NEVER answer it with a ranked leaderboard (ORDER BY ... DESC LIMIT N) and let the summary step average just the returned top-N rows - that silently drops everyone outside the top N and biases the result high. Verified against live data: the true average three-point percentage across all 47 qualifying (50+ attempt) Champions League 2024-25 players is ~37.3%; averaging only a top-20-by-percentage leaderboard instead gives a biased ~43.6% (a ~17% relative error) because it drops the other 27 qualifying players entirely.
- When that average also has a "minimum N attempts/games" qualifier, N is a SEASON TOTAL per player, never a per-row threshold - "minimum 50 attempts" means SUM(three_pa) across the player's whole season >= 50, not any single game having three_pa >= 50 (which is basically impossible and returns zero rows). Use a two-level aggregate: first GROUP BY player_name with HAVING SUM(x) >= N to get one row per qualifying player-season, THEN aggregate that qualifying set into the final single number. Example: WITH qualifying AS (SELECT player_name, SUM(three_pa) AS attempts, SUM(three_p) AS made FROM bronze.b_cl_boxscore WHERE <season dates> GROUP BY player_name HAVING SUM(three_pa) >= 50) SELECT SUM(made)*100.0/SUM(attempts) FROM qualifying.
- CRITICAL: never filter a team or player name with exact equality (team = 'Some Name') when the name comes from how a user naturally typed it in the question. Name casing is stored inconsistently across tables and is usually NOT natural title case - verified against live data: bronze team names in EL/EC/CL/BBL boxscore tables are stored ALL CAPS ('FC BAYERN MÜNCHEN BASKETBALL', not 'FC Bayern München Basketball'), and a plain = comparison against a naturally-typed name silently returns zero rows ("no data found") instead of erroring, which looks like a missing-data answer but is actually a wrong query. Always use a case-insensitive comparison for a user-supplied name filter: ILIKE, or UPPER(column) = UPPER('literal'). This applies to team names AND player names in any WHERE/JOIN condition, not just team names.

Table relationships:
{relationships}

Natural-language aliases:
- points, score, total points, scored = pts
- player, athlete = player_name
- team, club = player_team when using boxscore/player rows
- opponent = opponent when that column exists
- minutes played = minutes
- game, match = game_id or link depending on available columns
- image/logo = image_url or image columns when present
- shot location, shot position = wurfposition tables

Available tables and columns:
{tables}

Rules:
- Return exactly one PostgreSQL SELECT statement.
- If the question names a season, year, or date range, the query MUST include a matching WHERE filter on date (or saison, for gold tables) - see "Season and date filtering" above. Do not drop a time constraint just because it makes the query simpler.
- Use only tables from these schemas: {schemas}. Do not query the public schema unless metadata requires information_schema.
- Prefer schema-qualified table names, for example {default_schema}.table_name.
- Do not include markdown, comments, explanations, semicolons, INSERT, UPDATE, DELETE, DROP, ALTER, ATTACH, DETACH, PRAGMA, or CREATE.
- Prefer explicit joins and readable aliases.
- Add a LIMIT when returning row lists.
- For broad sample/list requests, use LIMIT 100 or less.
- Only prepare chart-style grouped/ranked data when the intent is draw. For table/schema/sample requests, return normal row or metadata results and do not force aggregation.
- For charts that use the minutes column, convert MM:SS text to decimal minutes with split_part(minutes, ':', 1)::numeric + split_part(minutes, ':', 2)::numeric / 60.0.
- For pts vs minutes scatter charts, filter minutes with minutes ~ '^[0-9]{{1,3}}:[0-9]{{2}}$' and exclude minutes = '00:00' unless the user asks for zero-minute rows.
- For top/ranking charts over positive statistics such as pts, prefer meaningful non-zero rows. If sorting ascending by an aggregate like SUM(pts), add a HAVING clause that removes zero totals unless the user explicitly asks for zero-value rows.
""".strip()

UNSAFE_SQL_PATTERN = re.compile(
    r"\b(insert|update|delete|drop|alter|attach|detach|pragma|create|replace|vacuum|reindex)\b",
    re.IGNORECASE,
)
SCHEMA_COLUMNS_SQL = """
SELECT table_schema, table_name, column_name, data_type
FROM information_schema.columns
WHERE table_schema = ANY(%s)
ORDER BY table_schema, table_name, ordinal_position
""".strip()
LIST_CONFIGURED_TABLES_SQL = """
SELECT table_schema, table_name
FROM information_schema.tables
WHERE table_schema = ANY(%s) AND table_type = 'BASE TABLE'
ORDER BY table_schema, table_name
LIMIT %s
""".strip()
COLUMNS_FOR_TABLE_SQL = """
SELECT table_schema, table_name, column_name, data_type, ordinal_position
FROM information_schema.columns
WHERE table_schema = %s AND table_name = %s
ORDER BY ordinal_position
""".strip()
COLUMN_COUNT_FOR_TABLE_SQL = """
SELECT table_schema, table_name, count(*)::integer AS column_count
FROM information_schema.columns
WHERE table_schema = %s AND table_name = %s
GROUP BY table_schema, table_name
""".strip()


class QueryEngineError(Exception):
    pass


class MissingApiKeyError(QueryEngineError):
    pass


class LlmProviderError(QueryEngineError):
    pass


class UnsafeSqlError(QueryEngineError):
    pass


def get_db_path() -> str:
    """Return the configured database URL for legacy call sites."""
    return get_database_url()


def configure_gemini(api_key: str | None) -> None:
    """Configure the Gemini client or raise when no key is available."""
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise MissingApiKeyError("Gemini API key is required. Add it in the sidebar or set GEMINI_API_KEY.")
    genai.configure(api_key=key)


def _gemini_model(model_name: str | None = None) -> genai.GenerativeModel:
    """Create a Gemini model instance using the configured default model."""
    return genai.GenerativeModel(model_name or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL))


def _ollama_generate(
    prompt: str,
    system: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> str:
    """Call an Ollama-compatible generate endpoint and return plain text."""
    root = (base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)).rstrip("/")
    endpoint = f"{root}/generate" if root.endswith("/api") else f"{root}/api/generate"
    key = api_key or os.getenv("OLLAMA_API_KEY")

    if "ollama.com" in root and not key:
        raise MissingApiKeyError("Ollama cloud requires an API key. Local Ollama does not.")

    payload = {
        "model": model_name or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise LlmProviderError(f"Ollama request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LlmProviderError("Ollama returned invalid JSON.") from exc

    if "error" in data:
        raise LlmProviderError(f"Ollama error: {data['error']}")
    return str(data.get("response", "")).strip()


def _openai_generate(
    prompt: str,
    system: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> str:
    """Call an OpenAI-compatible chat endpoint and return plain text.

    Used for the in-cluster LiteLLM proxy, which speaks the OpenAI protocol rather
    than Ollama's. Base URL already includes the version prefix, e.g.
    http://litellm.litellm.svc.cluster.local:4000/v1
    """
    root = (base_url or os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)).rstrip("/")
    endpoint = f"{root}/chat/completions"
    key = api_key or os.getenv("OPENAI_API_KEY")

    if not key:
        raise MissingApiKeyError(
            "An API key is required. Add it in the sidebar or set OPENAI_API_KEY "
            "(for LiteLLM this is a virtual key)."
        )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_name or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        "messages": messages,
        "stream": False,
    }

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise LlmProviderError(f"OpenAI-compatible request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LlmProviderError("OpenAI-compatible endpoint returned invalid JSON.") from exc

    if "error" in data:
        raise LlmProviderError(f"OpenAI-compatible error: {data['error']}")

    choices = data.get("choices") or []
    if not choices:
        raise LlmProviderError("OpenAI-compatible endpoint returned no choices.")
    return str(choices[0].get("message", {}).get("content", "")).strip()


def generate_text(
    prompt: str,
    system: str | None = None,
    provider: str | None = None,
    gemini_api_key: str | None = None,
    ollama_api_key: str | None = None,
    model_name: str | None = None,
    ollama_base_url: str | None = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
) -> str:
    """Generate text with the selected LLM provider."""
    selected = (provider or os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)).lower()
    if selected == "gemini":
        configure_gemini(gemini_api_key)
        parts = [prompt] if system is None else [system, prompt]
        response = _gemini_model(model_name).generate_content(parts)
        return (response.text or "").strip()
    if selected == "ollama":
        return _ollama_generate(
            prompt=prompt,
            system=system,
            model_name=model_name,
            base_url=ollama_base_url,
            api_key=ollama_api_key,
        )
    if selected == "openai":
        return _openai_generate(
            prompt=prompt,
            system=system,
            model_name=model_name,
            base_url=openai_base_url,
            api_key=openai_api_key,
        )
    raise LlmProviderError(f"Unsupported LLM provider: {selected}")


def extract_sql(text: str) -> str:
    """Extract a single SQL statement from raw model output."""
    cleaned = text.strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", cleaned, re.IGNORECASE | re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    cleaned = cleaned.rstrip(";").strip()
    return cleaned


def validate_sql(sql: str) -> str:
    """Validate that SQL is a single read-only SELECT statement (optionally CTE-prefixed)."""
    normalized = sql.strip()
    if not normalized:
        raise UnsafeSqlError("The LLM returned an empty SQL statement.")
    if not re.match(r"^(select|with)\b", normalized, flags=re.IGNORECASE):
        raise UnsafeSqlError("Only SELECT statements are allowed.")
    if ";" in normalized:
        raise UnsafeSqlError("Multiple SQL statements are not allowed.")
    if UNSAFE_SQL_PATTERN.search(normalized):
        raise UnsafeSqlError("The generated SQL contains a blocked keyword.")
    return normalized


def normalize_value(value: Any) -> Any:
    """Convert database-native values into UI-friendly Python values."""
    if isinstance(value, Decimal):
        return float(value)
    return value


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize every value in a database result row."""
    return {key: normalize_value(value) for key, value in row.items()}


def improve_sql_for_question(sql: str, question: str) -> str:
    """Apply deterministic SQL improvements for known prompt patterns."""
    normalized_question = question.lower()
    improved = sql
    if (
        "ascending" in normalized_question
        and "top" in normalized_question
        and "pts" in normalized_question
        and "sum(pts)" in improved.lower()
        and "having" not in improved.lower()
        and "zero" not in normalized_question
    ):
        improved = re.sub(
            r"(\bGROUP\s+BY\b.+?)(\bORDER\s+BY\b)",
            r"\1 HAVING SUM(pts) > 0 \2",
            improved,
            flags=re.IGNORECASE | re.DOTALL,
        )
    return validate_sql(improved)


def excluded_zero_totals(sql: str, question: str) -> bool:
    """Return whether ascending top-point SQL excluded zero totals."""
    normalized_question = question.lower()
    normalized_sql = sql.lower()
    return (
        "ascending" in normalized_question
        and "top" in normalized_question
        and "sum(pts)" in normalized_sql
        and "having sum(pts) > 0" in normalized_sql
        and "zero" not in normalized_question
    )


def classify_question(question: str) -> str:
    """Classify a prompt into the app's supported response intents."""
    normalized = question.lower()
    if has_draw_visualization_intent(question):
        return "draw"
    if "sql" in normalized and any(word in normalized for word in ("debug", "fix", "explain", "why", "error")):
        return "debug_sql"
    if any(
        phrase in normalized
        for phrase in (
            "list me all tables",
            "list all tables",
            "show tables",
            "welche tabellen",
            "alle tabellen",
            "zeige tabellen",
        )
    ):
        return "schema"
    if any(
        word in normalized
        for word in ("schema", "column", "columns", "tables", "spalte", "spalten", "tabelle")
    ):
        return "schema"
    if any(
        phrase in normalized
        for phrase in (
            "sample row",
            "sample rows",
            "show rows",
            "show me rows",
            "preview",
            "first rows",
            "raw rows",
            "beispielzeilen",
            "beispiel zeilen",
            "erste zeilen",
            "zeige zeilen",
            "rohdaten",
        )
    ):
        return "table"
    return "answer"


def has_draw_visualization_intent(question: str | None) -> bool:
    """Return whether a prompt explicitly asks for a visualization."""
    if not question:
        return False
    normalized = question.lower()
    draw_words = ("draw", "draws", "drawing", "draw me", "zeichne", "zeichnen", "male")
    visualization_words = (
        "chart",
        "diagram",
        "visualization",
        "visualize",
        "plot",
        "graph",
        "bar chart",
        "line chart",
        "scatter",
        "scatter plot",
        "histogram",
        # German. "diagramm" already matches "diagram" by substring, the rest
        # did not — a "Zeichne mir ein Balkendiagramm" fell through to the
        # plain answer intent and no chart was rendered.
        "diagramm",
        "grafik",
        "schaubild",
        "histogramm",
        "balkendiagramm",
        "liniendiagramm",
        "streudiagramm",
        "kreisdiagramm",
    )
    return any(word in normalized for word in draw_words + visualization_words)


def get_schema_prompt(db_path: str | None = None) -> str:
    """Build the LLM system prompt from live database schema metadata."""
    schemas = get_schemas()
    default_schema = get_default_schema()
    try:
        with connect(db_path) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(SCHEMA_COLUMNS_SQL, (schemas,))
                rows = [dict(row) for row in cur.fetchall()]
    except PostgresError as exc:
        message = str(exc).strip() or exc.__class__.__name__
        raise QueryEngineError(f"PostgreSQL error while reading schema: {message}") from exc

    if not rows:
        table_text = "(No tables found.)"
    else:
        table_columns: dict[str, list[str]] = {}
        for row in rows:
            qualified_name = f"{row['table_schema']}.{row['table_name']}"
            table_columns.setdefault(qualified_name, []).append(
                f"{row['column_name']} {row['data_type']}"
            )
        table_text = "\n".join(
            f"- {table}({', '.join(columns)})"
            for table, columns in table_columns.items()
        )

    relationships = build_relationship_notes(rows)
    return SCHEMA_PROMPT_TEMPLATE.format(
        # Resolved per request, not at import - a long-running pod would otherwise
        # keep answering "last season" against the date it happened to start on.
        today=date.today().isoformat(),
        schemas=", ".join(schemas),
        default_schema=default_schema,
        relationships=relationships,
        tables=table_text,
    )


def build_relationship_notes(rows: list[dict[str, Any]]) -> str:
    """Build domain relationship notes from discovered table names."""
    if not rows:
        return "- No table relationship hints are available because the schema is empty."

    table_columns: dict[str, set[str]] = {}
    for row in rows:
        table_columns.setdefault(f"{row['table_schema']}.{row['table_name']}", set()).add(row["column_name"])

    competition_names = {
        "b_el": "EuroLeague",
        "b_ec": "EuroCup",
        "b_cl": "Champions League",
        "b_bbl": "Basketball Bundesliga",
    }
    notes = []
    for prefix, competition in competition_names.items():
        family = sorted(table for table in table_columns if table.split(".", 1)[-1].startswith(f"{prefix}_"))
        if not family:
            continue
        notes.append(f"- {prefix}_* tables belong to {competition}: {', '.join(family)}.")

    notes.extend(
        [
            "- Within the same competition prefix, boxscore and playbyplay tables can often be compared by game/link-style columns when those columns exist.",
            "- Use player_name for player-level grouping. Use player_team for team-level grouping in boxscore-style rows.",
            "- Use player_info tables for roster/player metadata, boxscore tables for statistics, playbyplay tables for event logs, and wurfposition tables for shot locations.",
        ]
    )
    return "\n".join(notes)


def intent_instruction(intent: str) -> str:
    """Return prompt guidance for a classified question intent."""
    instructions = {
        "draw": "Intent: draw. Return chart-ready data: usually one label column and one or two numeric/date columns. Use GROUP BY/ORDER BY/LIMIT when useful.",
        "table": "Intent: table. Return raw rows for inspection. Do not aggregate unless the user explicitly asks for counts or totals. Always use a reasonable LIMIT.",
        "schema": "Intent: schema. Return metadata such as table names, columns, data types, or row counts.",
        "debug_sql": "Intent: debug_sql. Return SQL that helps inspect or validate the issue using read-only SELECTs.",
        "answer": "Intent: answer. Return the smallest result set needed to answer the question clearly.",
    }
    return instructions.get(intent, instructions["answer"])


def generate_sql(
    question: str,
    db_path: str | None = None,
    provider: str | None = None,
    gemini_api_key: str | None = None,
    ollama_api_key: str | None = None,
    model_name: str | None = None,
    ollama_base_url: str | None = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
) -> str:
    """Generate, extract, validate, and lightly improve SQL for a question."""
    intent = classify_question(question)
    retrieved_context = format_retrieved_context(question)
    rejected_context = format_rejected_context(question)
    prompt_parts = [intent_instruction(intent)]
    if retrieved_context:
        prompt_parts.append(retrieved_context)
    if rejected_context:
        prompt_parts.append(rejected_context)
    prompt_parts.append(f"Question: {question}")
    response = generate_text(
        prompt="\n\n".join(prompt_parts),
        system=get_schema_prompt(db_path=db_path),
        provider=provider,
        gemini_api_key=gemini_api_key,
        ollama_api_key=ollama_api_key,
        model_name=model_name,
        ollama_base_url=ollama_base_url,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
    )
    sql = extract_sql(response)
    return improve_sql_for_question(validate_sql(sql), question)


def repair_sql(
    question: str,
    failed_sql: str,
    error: str,
    db_path: str | None = None,
    provider: str | None = None,
    gemini_api_key: str | None = None,
    ollama_api_key: str | None = None,
    model_name: str | None = None,
    ollama_base_url: str | None = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
) -> str:
    """Ask the LLM to repair a failed SQL statement using the DB error."""
    intent = classify_question(question)
    retrieved_context = format_retrieved_context(question)
    rejected_context = format_rejected_context(question)
    prompt = f"""
The previous PostgreSQL query failed. Rewrite it as one valid, safe SELECT statement.

{intent_instruction(intent)}

{retrieved_context}

{rejected_context}

Question:
{question}

Failed SQL:
{failed_sql}

PostgreSQL error:
{error}
""".strip()
    response = generate_text(
        prompt=prompt,
        system=get_schema_prompt(db_path=db_path),
        provider=provider,
        gemini_api_key=gemini_api_key,
        ollama_api_key=ollama_api_key,
        model_name=model_name,
        ollama_base_url=ollama_base_url,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
    )
    sql = extract_sql(response)
    return validate_sql(sql)


def execute_sql(sql: str, db_path: str | None = None) -> list[dict[str, Any]]:
    """Execute validated SQL and return normalized rows with a row cap."""
    try:
        with connect(db_path) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql)
                rows = cur.fetchmany(MAX_QUERY_ROWS + 1)
    except PostgresError as exc:
        message = str(exc).strip() or exc.__class__.__name__
        if "connection" in message.lower():
            raise QueryEngineError("NBA Postgres database was not reachable. Start local Postgres first.") from exc
        raise QueryEngineError(f"PostgreSQL error: {message}") from exc
    if len(rows) > MAX_QUERY_ROWS:
        raise QueryEngineError(
            f"The query returned more than {MAX_QUERY_ROWS:,} rows. "
            "Ask a narrower question or include a LIMIT."
        )
    return [normalize_row(dict(row)) for row in rows]


def answer_schema_question(question: str, db_path: str | None = None) -> tuple[str, str, list[dict[str, Any]]] | None:
    """Answer simple table-list questions without calling the LLM."""
    normalized = question.lower()
    column_answer = answer_column_metadata_question(question, db_path=db_path)
    if column_answer is not None:
        return column_answer

    asks_for_tables = "table" in normalized and any(
        phrase in normalized
        for phrase in ("list", "show", "what", "which", "all")
    )
    if "first table" not in normalized and not asks_for_tables:
        return None

    schemas = get_schemas()
    try:
        with connect(db_path) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(LIST_CONFIGURED_TABLES_SQL, (schemas, 1 if "first table" in normalized else 10000))
                rows = cur.fetchall()
    except PostgresError as exc:
        message = str(exc).strip() or exc.__class__.__name__
        raise QueryEngineError(f"PostgreSQL error: {message}") from exc

    rows = [dict(row) for row in rows]
    if not rows:
        return (
            f"I could not find any tables in the configured PostgreSQL schemas: `{', '.join(schemas)}`.",
            f"SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema = ANY({schemas!r}) ORDER BY table_schema, table_name",
            rows,
        )

    rendered_sql = (
        "SELECT table_schema, table_name\n"
        "FROM information_schema.tables\n"
        f"WHERE table_schema = ANY(ARRAY{schemas!r}) AND table_type = 'BASE TABLE'\n"
        "ORDER BY table_schema, table_name"
    )
    if "first table" not in normalized:
        table_list = ", ".join(f"`{row['table_schema']}.{row['table_name']}`" for row in rows)
        return f"Die konfigurierten Schemas enthalten diese Tabellen: {table_list}.", rendered_sql, rows

    table_name = f"{rows[0]['table_schema']}.{rows[0]['table_name']}"
    answer = (
        f"The first table in the configured PostgreSQL schemas is `{table_name}`. "
        f"`{table_name}` is a table name, not one specific team. "
        "If you meant the first row inside that table, ask: "
        "`Which team is the first row in the teams table?`"
    )
    return answer, rendered_sql, rows


def answer_column_metadata_question(
    question: str,
    db_path: str | None = None,
) -> tuple[str, str, list[dict[str, Any]]] | None:
    """Answer specific column-count and column-list metadata questions."""
    normalized = question.lower()
    if "column" not in normalized and "columns" not in normalized:
        return None

    table_reference = table_name_from_question(question)
    if not table_reference:
        return None

    resolved_table = resolve_table_reference(table_reference, db_path=db_path)
    if not resolved_table:
        return None

    table_schema, table_name = resolved_table
    wants_count = any(phrase in normalized for phrase in ("how many", "count", "number of"))
    query_sql = COLUMN_COUNT_FOR_TABLE_SQL if wants_count else COLUMNS_FOR_TABLE_SQL

    try:
        with connect(db_path) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query_sql, (table_schema, table_name))
                rows = [dict(row) for row in cur.fetchall()]
    except PostgresError as exc:
        message = str(exc).strip() or exc.__class__.__name__
        raise QueryEngineError(f"PostgreSQL error: {message}") from exc

    rendered_sql = (
        query_sql.replace("%s", f"'{table_schema}'", 1)
        .replace("%s", f"'{table_name}'", 1)
    )
    qualified_name = f"{table_schema}.{table_name}"
    if not rows:
        return f"Für `{qualified_name}` konnte ich keine Spalten finden.", rendered_sql, rows

    if wants_count:
        count = rows[0]["column_count"]
        return f"`{qualified_name}` has {count:,} columns.", rendered_sql, rows

    return f"Das sind die Spalten von `{qualified_name}`.", rendered_sql, rows


def summarize_answer(
    question: str,
    sql: str,
    rows: list[dict[str, Any]],
    provider: str | None = None,
    gemini_api_key: str | None = None,
    ollama_api_key: str | None = None,
    model_name: str | None = None,
    ollama_base_url: str | None = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
) -> str:
    """Summarize SQL result rows into a user-facing answer."""
    if not rows:
        return "Keine Daten für diese Frage gefunden. Versuch es mit einer engeren Tabelle, einem Zeitraum, Team oder Spieler."

    intent = classify_question(question)
    if intent == "draw":
        return chart_answer(question, sql, rows)

    if intent in {"table", "schema"}:
        return table_answer(question, sql, rows)

    prompt = f"""
Answer the user's basketball analytics question using only the SQL result rows below.

Question:
{question}

SQL:
{sql}

Rows:
{rows[:100]}

Give a concise, formatted answer IN THE SAME LANGUAGE AS THE QUESTION. Mention if the result is limited by returned rows or a SQL LIMIT.
""".strip()
    response = generate_text(
        prompt=prompt,
        provider=provider,
        gemini_api_key=gemini_api_key,
        ollama_api_key=ollama_api_key,
        model_name=model_name,
        ollama_base_url=ollama_base_url,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
    )
    return response or "I found matching rows, but the LLM returned an empty answer."


def is_chart_request(question: str) -> bool:
    """Return whether a question requests a chart."""
    return has_draw_visualization_intent(question)


def table_answer(question: str, sql: str, rows: list[dict[str, Any]]) -> str:
    """Create a concise answer for table and schema inspection results."""
    row_count = len(rows)
    columns = list(rows[0].keys()) if rows else []
    table_name = referenced_table_name(sql)
    target = f" aus `{table_name}`" if table_name else ""
    if "column" in question.lower() or "schema" in question.lower():
        return f"Schema-Details{target}: {row_count} Zeilen."
    return f"{row_count} Zeilen{target}, {len(columns)} Spalten."


def referenced_table_name(sql: str) -> str | None:
    """Extract the first referenced table name from a SQL FROM clause."""
    match = re.search(
        r'\bfrom\s+(?:"?([a-zA-Z_][\w]*)"?\.)?"?([a-zA-Z_][\w]*)"?',
        sql,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    schema, table = match.groups()
    return f"{schema}.{table}" if schema else table


def table_name_from_question(question: str) -> str | None:
    """Extract a schema-qualified or unqualified table name from a prompt."""
    # The optional schema part trails the name rather than leading it. Written
    # as "(?:name\.)?name" the two \w* runs compete for the same characters, so
    # a long run of word characters backtracks quadratically before failing.
    # Anchoring the optional half behind a literal dot removes the ambiguity;
    # the set of accepted strings is unchanged.
    for pattern in (
        r"\b(?:from|in\s+table|table|in)\s+([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)?)",
        r"\b([a-zA-Z_]\w*\.[a-zA-Z_]\w*)",
    ):
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def resolve_table_reference(table_reference: str, db_path: str | None = None) -> tuple[str, str] | None:
    """Resolve a prompt table reference against configured schemas."""
    schemas = get_schemas()
    if "." in table_reference:
        table_schema, table_name = table_reference.split(".", 1)
        if table_schema not in schemas:
            return None
        where_clause = "table_schema = %s AND table_name = %s"
        params: tuple[Any, ...] = (table_schema, table_name)
    else:
        where_clause = "table_schema = ANY(%s) AND table_name = %s"
        params = (schemas, table_reference)

    try:
        with connect(db_path) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"""
                    SELECT table_schema, table_name
                    FROM information_schema.tables
                    WHERE {where_clause}
                      AND table_type = 'BASE TABLE'
                    ORDER BY array_position(%s, table_schema), table_name
                    LIMIT 1
                    """,
                    (*params, schemas),
                )
                row = cur.fetchone()
    except PostgresError as exc:
        message = str(exc).strip() or exc.__class__.__name__
        raise QueryEngineError(f"PostgreSQL error: {message}") from exc

    return (row["table_schema"], row["table_name"]) if row else None


def answer_known_chart_question(question: str, db_path: str | None = None) -> tuple[str, str, list[dict[str, Any]]] | None:
    """Answer deterministic chart prompts that need domain-specific SQL."""
    normalized = question.lower()
    if classify_question(question) != "draw":
        return None
    if "scatter" not in normalized or "pts" not in normalized or "minutes" not in normalized:
        return None

    requested_table = table_name_from_question(question)
    # PostgreSQL identifiers stop at 63 bytes, so anything longer cannot name a
    # real table - rejecting it up front bounds the work regardless of pattern.
    if not requested_table or len(requested_table) > 127:
        return None
    if not re.fullmatch(r"[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)?", requested_table):
        return None

    if "." in requested_table:
        schema, table = requested_table.split(".", 1)
    else:
        schema = get_default_schema()
        table = requested_table
    if schema not in get_schemas():
        return None
    sql = pg_sql.SQL(
        """
        SELECT
            split_part(minutes, ':', 1)::numeric + split_part(minutes, ':', 2)::numeric / 60.0 AS minutes,
            pts,
            player_name,
            player_team
        FROM {}.{}
        WHERE minutes ~ '^[0-9]{{1,3}}:[0-9]{{2}}$'
          AND pts IS NOT NULL
          AND minutes <> '00:00'
        LIMIT 500
        """
    ).format(pg_sql.Identifier(schema), pg_sql.Identifier(table)).as_string()
    sql = "\n".join(line.strip() for line in sql.strip().splitlines())
    rows = execute_sql(validate_sql(sql), db_path=db_path)
    answer = (
        f"Diagrammfertiges Ergebnis mit {len(rows)} Zeilen. "
        "`minutes` wurde von MM:SS in Dezimalminuten umgerechnet, `00:00`-Zeilen sind ausgeschlossen."
    )
    return answer, sql, rows


def chart_answer(question: str, sql: str, rows: list[dict[str, Any]]) -> str:
    """Create a concise answer for chart-ready result sets."""
    row_count = len(rows)
    limited = "limit" in sql.lower()
    notes = []
    if limited:
        notes.append("Das SQL-Ergebnis ist durch ein LIMIT begrenzt.")
    if excluded_zero_totals(sql, question):
        notes.append("Null-Punkte-Summen sind ausgeschlossen, damit das aufsteigende Diagramm bei echten Scorern beginnt.")
    suffix = " " + " ".join(notes) if notes else ""
    return f"Diagrammfertiges Ergebnis mit {row_count} Zeilen.{suffix}"


def answer_question(
    question: str,
    db_path: str | None = None,
    provider: str | None = None,
    gemini_api_key: str | None = None,
    ollama_api_key: str | None = None,
    model_name: str | None = None,
    ollama_base_url: str | None = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Answer a user question by routing, generating SQL, and summarizing rows."""
    schema_answer = answer_schema_question(question, db_path=db_path)
    if schema_answer is not None:
        return schema_answer

    known_chart_answer = answer_known_chart_question(question, db_path=db_path)
    if known_chart_answer is not None:
        return known_chart_answer

    sql = generate_sql(
        question,
        db_path=db_path,
        provider=provider,
        gemini_api_key=gemini_api_key,
        ollama_api_key=ollama_api_key,
        model_name=model_name,
        ollama_base_url=ollama_base_url,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
    )
    try:
        rows = execute_sql(sql, db_path=db_path)
    except QueryEngineError as exc:
        repaired_sql = repair_sql(
            question,
            failed_sql=sql,
            error=str(exc),
            db_path=db_path,
            provider=provider,
            gemini_api_key=gemini_api_key,
            ollama_api_key=ollama_api_key,
            model_name=model_name,
            ollama_base_url=ollama_base_url,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
        )
        rows = execute_sql(repaired_sql, db_path=db_path)
        sql = repaired_sql
    answer = summarize_answer(
        question,
        sql,
        rows,
        provider=provider,
        gemini_api_key=gemini_api_key,
        ollama_api_key=ollama_api_key,
        model_name=model_name,
        ollama_base_url=ollama_base_url,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
    )
    return answer, sql, rows
