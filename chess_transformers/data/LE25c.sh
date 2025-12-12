# Run these commands in the folder with the downloaded Lichess Elite PGN files from Sep 2013 to Aug 2025

# Filter games that ended in a checkmate
JOBS=16                       
find . -name "lichess_elite_20*.pgn" | sort > all.txt
split -n l/"$JOBS" all.txt split-
for f in split-*; do
    {
        out="${f}.pgn"       
        pgn-extract -f "$f" --checkmate -s -o "$out"
    } &
done
wait
cat split-*.pgn > filtered_games.pgn
rm -- split-* all.txt

# Shuffle games
python - <<'PY'
import random, regex, pathlib, sys
random.seed(1234)
text   = pathlib.Path('filtered_games.pgn').read_text()
games  = regex.split(r'\n{2,}(?=\[Event)', text)
random.shuffle(games)
pathlib.Path('shuffled_filtered_games.pgn').write_text('\n\n'.join(games))
print(f'{len(games):,} games shuffled.')
PY

# Split into smaller chunks with 500k games each, for ease of handling in python without OOM issues
awk -v RS='\n\\[Event'  -v N=500000 '
  NR % N == 1 {                # starting a new output file?
      if (out) close(out)
      out = sprintf("chunk_%03d.pgn", ++c)
  }
  {                            # $0 = record without leading [Event
      if (FNR == 1 && NR == 1) # very first record overall
          print "[Event" $0     > out
      else
          print "\n[Event" $0   > out
  }
' shuffled_filtered_games.pgn

# Get the board at each position in each game in FEN (Forsyth–Edwards Notation)
for f in chunk_00{1..9}.pgn; do
{
    base=${f%.pgn}
    pgn-extract -s -Wfen "$f" \
                --notags --noresults \
                --output "${base}.fens"
} &
done
wait

# Get the corresponding moves in UCI format
for f in chunk_00{1..9}.pgn; do
{
    base=${f%.pgn}
    pgn-extract -s -Wlalg "$f" \
                --notags --nomovenumbers --nochecks -w7 \
                --output "${base}.moves"
} &
done
wait

# Remove files we no longer need
rm *.pgn