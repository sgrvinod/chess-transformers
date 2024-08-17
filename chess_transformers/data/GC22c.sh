# Run these commands in the folder with the PGN file corresponding to the dataset here: https://www.kaggle.com/datasets/dimitrioskourtikakis/gm-games-chesscom
# This is purported to contain chess games played by Grandmaster titled players on chess.com up to July 2022

# Remove duplicate games

pgn-extract -ddupes.pgn -ounique.pgn GC22.pgn

# "4811076 games matched out of 4811076." <- This is not the number of unique files, just those checked

# Remove files we no longer need

rm dupes.pgn GC22.pgn

# Filter games that ended in a checkmate

pgn-extract --checkmate unique.pgn --output filtered_games.pgn

# "593693 games matched out of 4178495."

# Remove files we no longer need

rm unique.pgn

# Shuffle games

python3 -c "import regex, random; random.seed(1234); text = open('filtered_games.pgn', 'r', encoding='latin-1').read(); games = regex.split(r'\n{2,}(?=\[Event)', text); random.shuffle(games); print('{} games shuffled.'.format(len(games))); open('shuffled_filtered_games.pgn', 'w').write('\n\n'.join(games));"

# "593693 games shuffled."

# Split into smaller chunks with 500k games each, for ease of handling in python without OOM issues

pgn-extract -#500000 shuffled_filtered_games.pgn

# "593693 games matched out of 593693."

# Remove files we no longer need

rm filtered_games.pgn shuffled_filtered_games.pgn

# Get the board at each position in each game in FEN (Forsyth–Edwards Notation)

for i in {1..2}
do
    pgn-extract -Wfen $i.pgn --notags --noresults --output $i.fens
done

# "500000 games matched out of 500000."
# "93693 games matched out of 93693."

# Get the corresponding moves in UCI format

for i in {1..2}
do
    pgn-extract -Wlalg $i.pgn --notags --nomovenumbers --nochecks -w7 --output $i.moves
done

# "500000 games matched out of 500000."
# "93693 games matched out of 93693."

# Remove files we no longer need

rm *.pgn