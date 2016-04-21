
## SEQ2SEQ (translate.py)

pretrain

        If mentionned we use pretrain() function instead of train()
        
        
encoder_num

        Num of the encoders to include in the model(s) during pretrain
        Lets a model with 4 encoders has been created from 0 to 3 with parameters --src_ext en,fr,de,it
        --model_name m1,m2,m3,m4
        Now we want to pretrain fr and it only, then we will use --pretrain --src_ext fr,it --encoder_num 1,3 
        --model_name m1,m2
        If ignored, pretrain use encoder 0,1...,n.
        

model_name

        Name of the models
        Models are sharing variables belonging to the session. But there are two variables that are specific to each
        model : learning_rate and global_step (even though this last one may be useless, to confirm). To avoid conflict
        in the namespace, we give a name to every model created for these two var. Must match src_ext.
        Lets say src_ext is fr,de then model_name must have at least to name (ex : m1,m2)


embedding_prefix

        Name of the embedding file for the languages (encoder and decoder)
        Lets say you have three src_ext de,fr,en and one trg_ext it with --embedding embed_file, there should exist 
        embed_file.fr, embed_file.de, embed_file.en and embed_file.it. If one of this file doesnt exist, the model initiliaze the 
        embedding matrix for the language as usual.
    
fix_embedding

        A list of 0 or 1 declaring wether an embedded given in parameters should be fixed (not trained) or not along with the model
        Lets say src_ext fr,de and trg_ext en and there exists an embed.en we want to train and an embed.de we dont want to train.
        We do --embedding embed and --fix_embedding (1/0), 1, 0. First parameter of fix_embedding doesnt matter
        because embed.fr doesnt exist, 3rd is for decoder.
        If embedding_prefix is given but not fix_embedding, then embeddings are trained by default


lookup_dict
	
	Name of the lookup dictionnary of the unsupervised align. If mentionned, decode_sentence func will try to repace UNK tokens.


## PREPROCESSING FOR UNK WORDS (prepare_data.py)

align

	If mentionned, creates unsupervised word aligned "fast align". See https://github.com/clab/fast_align
    
dict_val_select
	
	Number of occurence for a pair of words in the align file to be selected in the lookup dictionnary. Default : 100

fast_align_loc
	
	Location of the fast_align binary in scripts. Default : "fast_align" (scripts/fast_align)
        
fast_align_iter

	Number of iterations in fast align learning. Default : 5            
      
	
