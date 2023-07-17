# Run these commands in the folder with the downloaded Lichess Elite PGN files from 2013-2021

# Create a file with the names of all PGN files we want to process

find . -name "lichess_elite_20*.pgn" | sort > filenames.txt

# Combine all these files into a single file

pgn-extract -ffilenames.txt --output combined.pgn

# Since pgn-extract also checks games, it finds some inconsistencies:
#
# "Warning: Result of 1-0 is inconsistent with stalemate in
# old_friend - Suyud28 Rated Blitz game 2020.10.30 
# File ./lichess_elite_2020-10.pgn: Line number: 10165543
# Warning: Result of 0-1 is inconsistent with stalemate in
# KingPinLopez - dMachin Rated Blitz game 2021.03.12 
# File ./lichess_elite_2021-03.pgn: Line number: 5525339
# Warning: Result of 0-1 is inconsistent with stalemate in
# Darcy95 - Lunaticxxx Rated Blitz game 2021.04.01 
# File ./lichess_elite_2021-04.pgn: Line number: 144696"
# 
# We can ignore them

# There's more than 16 million games! "16042102 games matched out of 16042102." 

# Add the "--quiet" parameter if you want to suppress game-by-game progress output with pgn-extract...
# ...but this will also suppress game count at the end
# It will not suppress warnings or errors

# Remove files we no longer need

rm lichess_elite_20*.pgn filenames.txt

# Create a file with a time control filter (>= 5 minutes base time per player)...
# ...because this filter has to be used from a file, and not the commandline, as far as I can tell

echo 'TimeControl >= "300"' > tags.txt

# Apply filter and store filtered games in a new file

pgn-extract -t tags.txt combined.pgn --output time_control_gte_5m.pgn

# "1661885 games matched out of 16042102."

# Remove files we no longer need

rm combined.pgn tags.txt 

# Filter games that ended in a checkmate

pgn-extract --checkmate time_control_gte_5m.pgn --output checkmate_gte5m.pgn

# "214454 games matched out of 1661885."

# Filter games that ended in a stalemate

pgn-extract --stalemate time_control_gte_5m.pgn --output stalemate_gte5m.pgn

# "8028 games matched out of 1661885."

# Filter games that ended in a draw by the 50-move rule

pgn-extract --fifty time_control_gte_5m.pgn --output fifty_move_draw_gte5m.pgn

# "2517 games matched out of 1661885."

# Filter games that ended in a draw by the repetition rule

pgn-extract --repetition time_control_gte_5m.pgn --output repetition_draw_gte5m.pgn

# "83257 games matched out of 1661885."

# Remove files we no longer need

rm time_control_gte_5m.pgn

# Combined these filtered games into a new file

pgn-extract checkmate_gte5m.pgn stalemate_gte5m.pgn fifty_move_draw_gte5m.pgn repetition_draw_gte5m.pgn --output filtered_games.pgn

# A little over 300k games in total! "308256 games matched out of 308256."

# Remove files we no longer need

rm checkmate_gte5m.pgn stalemate_gte5m.pgn fifty_move_draw_gte5m.pgn repetition_draw_gte5m.pgn

# Split into smaller files with 80k games each, for ease of handling in python without OOM issues

pgn-extract -#80000 filtered_games.pgn

# "308256 games matched out of 308256."

# Remove files we no longer need

rm filtered_games.pgn










