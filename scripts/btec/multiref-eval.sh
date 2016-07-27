

# greedy test1
python2 -m translate data/btec/ models/ensemble/ --checkpoints models/ensemble/model_5/checkpoints.fr_en/best --reset --size 256 --use-lstm --bidir --num-layers 2 --vocab-size 10000 --beam-size 1 -v --decode data/btec/test1 --output results/single/test1.greedy.mt --gpu-id 0

# greedy test2
python2 -m translate data/btec/ models/ensemble/ --checkpoints models/ensemble/model_5/checkpoints.fr_en/best --reset --size 256 --use-lstm --bidir --num-layers 2 --vocab-size 10000 --beam-size 1 -v --decode data/btec/test2 --output results/single/test2.greedy.mt --gpu-id 1

# greedy dev
python2 -m translate data/btec/ models/ensemble/ --checkpoints models/ensemble/model_5/checkpoints.fr_en/best --reset --size 256 --use-lstm --bidir --num-layers 2 --vocab-size 10000 --beam-size 1 -v --decode data/btec/dev --output results/single/dev.greedy.mt --gpu-id 0

# ensemble + LM test1
python2 -m translate data/btec/ models/ensemble/ --checkpoints models/ensemble/model_{1,2,3,4,5}/checkpoints.fr_en/best --ensemble --use-lm --lm-order 3 --reset --size 256 --use-lstm --bidir --num-layers 2 --vocab-size 10000 --beam-size 8 -v --decode data/btec/test1 --output results/ensemble/test1.lm.mt --gpu-id 0

# ensemble + LM test2
python2 -m translate data/btec/ models/ensemble/ --checkpoints models/ensemble/model_{1,2,3,4,5}/checkpoints.fr_en/best --ensemble --use-lm --lm-order 3 --reset --size 256 --use-lstm --bidir --num-layers 2 --vocab-size 10000 --beam-size 8 -v --decode data/btec/test2 --output results/ensemble/test2.lm.mt --gpu-id 1

# ensemble + LM dev
python2 -m translate data/btec/ models/ensemble/ --checkpoints models/ensemble/model_{1,2,3,4,5}/checkpoints.fr_en/best --ensemble --use-lm --lm-order 3 --reset --size 256 --use-lstm --bidir --num-layers 2 --vocab-size 10000 --beam-size 8 -v --decode data/btec/dev --output results/ensemble/dev.lm.mt --gpu-id 0



# scoring:

bin/scoring/score.rb --refs-laced data/btec/test1.en --hyp-detok results/ensemble/test1.lm.mt --print


