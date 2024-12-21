# Run these commands in the folder with the downloaded PGN file containing games from Lumbra's Giga Base downloaded in November 2024
# Note: I downloaded the SCID 5 database (Version: 2024-11-05) and exported its entirety (except the single game mentioned below) to a single PGN file called "LG24.pgn"
# I excluded the sole game played by the player 'Asker-B. Ãstenstad' on White because this game caused pgn-extract processes to stop indefinitely, very likely due to an encoding mismatch
# I'm unsure what encoding the PGN file was exported in by SCID, but below, I use 'latin-1' because this works well with pgn-extract

# Remove duplicate games

pgn-extract -ddupes.pgn -ounique.pgn LG24.pgn

# "14905185 games matched out of 14905398." <- This is not the number of unique files, just those checked

# Remove files we no longer need

rm dupes.pgn LG24.pgn

# Remove games with fewer than 20 plies (or half-moves)

pgn-extract --minply 20 unique.pgn --output not_short.pgn

# "14686699 games matched out of 14803111."

# Remove files we no longer need

rm unique.pgn

# Create a file with an ELO filter (either player must have >=2300)...
# ...because this filter has to be used from a file, and not the commandline, as far as I can tell

echo 'Elo >= "2300"' > tags.txt

# Apply filter and store filtered games in a new file

pgn-extract -t tags.txt not_short.pgn --output elo_gte_2300.pgn

# "9662135 games matched out of 14686699."

# Remove files we no longer need

rm not_short.pgn

# Create a file with a date filter (games after 2000)...
# ...because this filter has to be used from a file, and not the commandline, as far as I can tell

echo 'Date >= "2000"' > tags.txt

# Apply filter and store filtered games in a new file

pgn-extract -t tags.txt elo_gte_2300.pgn --output after_2000.pgn

# "8790297 games matched out of 9662135."

# Remove files we no longer need

rm elo_gte_2300.pgn

# Filter decisive games

pgn-extract -Tr1-0 -Tr0-1 after_2000.pgn --output filtered_games.pgn

# "7059375 games matched out of 8790297."

# Remove files we no longer need

rm after_2000.pgn

# Shuffle games

python3 -c "import regex, random; random.seed(1234); text = open('filtered_games.pgn', 'r', encoding='latin-1').read(); games = regex.split(r'\n{2,}(?=\[Event)', text); random.shuffle(games); print('{} games shuffled.'.format(len(games))); open('shuffled_filtered_games.pgn', 'w', encoding='latin-1').write('\n\n'.join(games));"

# "7059375 games shuffled."

# Split into smaller chunks with 500k games each, for ease of handling in python without OOM issues

pgn-extract -#500000 shuffled_filtered_games.pgn

# "7059375 games matched out of 7059375."

# Remove files we no longer need

rm filtered_games.pgn shuffled_filtered_games.pgn

# Get the board at each position in each game in FEN (Forsyth–Edwards Notation)

for i in {1..15}
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
# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "59375 games matched out of 59375."

# Get the corresponding moves in UCI format

for i in {1..15}
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
# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "500000 games matched out of 500000."
# "59375 games matched out of 59375."

# Remove files we no longer need

rm *.pgn