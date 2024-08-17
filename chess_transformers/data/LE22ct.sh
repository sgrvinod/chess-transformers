# Run these commands in the folder with the downloaded Lichess Elite PGN files from Sep 2013 to Dec 2022

# Create a file with the names of all PGN files we want to process

find . -name "lichess_elite_20*.pgn" | sort > filenames.txt

# Combine all these files into a single file

pgn-extract -ffilenames.txt --output combined.pgn

# There's more than 20 million games! "20241368 games matched out of 20241368." 

# Remove files we no longer need

rm lichess_elite_20*.pgn filenames.txt

# Create a file with a time control filter (>= 5 minutes base time per player)...
# ...because this filter has to be used from a file, and not the commandline, as far as I can tell

echo 'TimeControl >= "300"' > tags.txt

# Apply filter and store filtered games in a new file

pgn-extract -t tags.txt combined.pgn --output time_control_gte_5m.pgn

# "2073780 games matched out of 20241368."

# Remove files we no longer need

rm combined.pgn tags.txt 

# Filter games that ended in a checkmate

pgn-extract --checkmate time_control_gte_5m.pgn --output filtered_games.pgn

# "274794 games matched out of 2073780."

# Remove files we no longer need

rm time_control_gte_5m.pgn

# Shuffle games

python3 -c "import regex, random; random.seed(1234); text = open('filtered_games.pgn', 'r').read(); games = regex.split(r'\n{2,}(?=\[Event)', text); random.shuffle(games); print('{} games shuffled.'.format(len(games))); open('shuffled_filtered_games.pgn', 'w').write('\n\n'.join(games));"

# "274794 games shuffled."

# Split into smaller chunks with 500k games each, for ease of handling in python without OOM issues

pgn-extract -#500000 shuffled_filtered_games.pgn

# "274794 games matched out of 274794."

# Remove files we no longer need

rm filtered_games.pgn shuffled_filtered_games.pgn

# Get the board at each position in each game in FEN (Forsyth–Edwards Notation)

pgn-extract -Wfen 1.pgn --notags --noresults --output 1.fens

# "274794 games matched out of 274794."

# Get the corresponding moves in UCI format

pgn-extract -Wlalg 1.pgn --notags --nomovenumbers --nochecks -w7 --output 1.moves

# "274794 games matched out of 274794."

# Remove files we no longer need

rm *.pgn










