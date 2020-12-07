#import functions from other files
import nbimporter
import ReadBedFiles
import Region2vec



import sys



if __name__ == "__main__":
   
    args = sys.argv
    
    path_bed_files = args[1]
    universeFile_path = args[2]
    numberofCores = int(args[3])
    dimension = int(args[4])
    min_count = int(args[5])
    shuffle_repeat = int(args[6])
    window_size = int(args[7])


    # Read bed files 
    term_doc_matrix, segmentation_df = ReadBedFiles.readFiles2Vector(path_bed_files, universeFile_path, numberofCores, 100)
    print('Reading files Done')

    # convert term-doc matrix to Corpus
    documents = ReadBedFiles.convertMat2document(term_doc_matrix, segmentation_df)
    print('Conversion Done')

    # Shuffle documents for training
    shuffeled_documents = Region2vec.shuffling(documents, shuffle_repeat)
    print('Shuffling Done')

    # Train word2Vec model
    model = Region2vec.trainWord2Vec(shuffeled_documents, window_size, dimension, min_count)
    
    print('Training Done')
    model.save('w2v_shfl{}_dim{}_mnct{}_winsize{}.model'.format(shuffle_repeat, dimension, min_count, window_size))
    print(len(model.wv.vocab))