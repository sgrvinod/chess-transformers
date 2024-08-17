# Run these commands in the folder with the downloaded PGN files containing Master level games from PGN mentor, TWIC, and Caissabase in Dec 2023

# Create a file with the names of all PGN files we want to process

find . -name "*.pgn" | sort > filenames.txt

# Combine all these files into a single file

pgn-extract -ffilenames.txt --output combined.pgn

# "11081724 games matched out of 11082125."

# Remove files we no longer need

find . ! -name 'combined.pgn' -type f -exec rm -f {} +

# Remove duplicate games

pgn-extract -ddupes.pgn -ounique.pgn combined.pgn

# "11081712 games matched out of 11081712." <- This is not the number of unique files, just those checked

# Remove files we no longer need

rm dupes.pgn combined.pgn

# Filter decisive games

pgn-extract -Tr1-0 -Tr0-1 unique.pgn --output filtered_games.pgn

# "3739604 games matched out of 5213634."

# Remove files we no longer need

rm unique.pgn

# Shuffle games

python3 -c "import regex, random; random.seed(1234); text = open('filtered_games.pgn', 'r', encoding='latin-1').read(); games = regex.split(r'\n{2,}(?=\[Event)', text); random.shuffle(games); print('{} games shuffled.'.format(len(games))); open('shuffled_filtered_games.pgn', 'w').write('\n\n'.join(games));"

# "3739604 games shuffled."

# Split into smaller chunks with 500k games each, for ease of handling in python without OOM issues

pgn-extract -#500000 shuffled_filtered_games.pgn

# "3739604 games matched out of 3739604."

# Remove files we no longer need

rm filtered_games.pgn shuffled_filtered_games.pgn

# Get the board at each position in each game in FEN (Forsyth–Edwards Notation)

for i in {1..8}
do
    pgn-extract -Wfen $i.pgn --notags --noresults --output $i.fens
done

# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "239604 games matched out of 239604."

# Get the corresponding moves in UCI format

for i in {1..8}
do
    pgn-extract -Wlalg $i.pgn --notags --nomovenumbers --nochecks -w7 --output $i.moves
done

# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "239604 games matched out of 239604."

# Remove files we no longer need

rm *.pgn