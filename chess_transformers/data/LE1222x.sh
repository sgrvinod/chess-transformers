# Run these commands in the folder with the downloaded Lichess Elite PGN files from Sep 2013 to Dec 2022

# Create a file with the names of all PGN files we want to process

find . -name "lichess_elite_20*.pgn" | sort > filenames.txt

# Combine all these files into a single file

pgn-extract -ffilenames.txt --output combined.pgn

# There's more than 20 million games! "20241368 games matched out of 20241368." 

# Remove files we no longer need

rm lichess_elite_20*.pgn filenames.txt

# Filter games that ended in a checkmate

pgn-extract --checkmate combined.pgn --output filtered_games.pgn

# "2751394 games matched out of 20241368."

# Remove files we no longer need

rm combined.pgn

# Shuffle games

python3 -c "import regex, random; random.seed(1234); text = open('filtered_games.pgn', 'r').read(); games = regex.split(r'\n{2,}(?=\[Event)', text); random.shuffle(games); print('{} games shuffled.'.format(len(games))); open('shuffled_filtered_games.pgn', 'w').write('\n\n'.join(games));"

# "2751394 games shuffled."

# Split into smaller chunks with 500k games each, for ease of handling in python without OOM issues

pgn-extract -#500000 shuffled_filtered_games.pgn

# "2751394 games matched out of 20241368."

# Remove files we no longer need

rm filtered_games.pgn shuffled_filtered_games.pgn

# Get the board at each position in each game in FEN (Forsyth–Edwards Notation)

for i in {1..6}
do
    pgn-extract -Wfen $i.pgn --notags --noresults --output $i.fens
done

# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "251394 games matched out of 251394."

# Get the corresponding moves in UCI format

for i in {1..6}
do
    pgn-extract -Wlalg $i.pgn --notags --nomovenumbers --nochecks -w7 --output $i.moves
done

# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "251394 games matched out of 251394."

# Remove files we no longer need

rm *.pgn










