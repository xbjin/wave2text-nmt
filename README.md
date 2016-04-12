

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


embedding
        Name of the embedding file for the languages. 
        Lets say you have three src_ext : de,fr,en and --embedding embed_file, there should exist 
        embed_file.fr, embed_file.de and embed_file.en. If one of this file doesnt exist, the model initiliaze the 
        embedding matrix for the language as usual.
    
embedding_train
        A list of True or False declaring wether an embedded given in parameters should be trained or not along with the model
        Lets say src_ext are fr,de and there exists an embed.de that we want to pass to the model and train.
        We do --embedding embed and --embedding_train (None/True/False), True. First parameter of embedding_train doesnt matter
        because embed.fr doesnt exist.
        If embedding parameter is given but not embedding_train, then embeddings are trained by default
