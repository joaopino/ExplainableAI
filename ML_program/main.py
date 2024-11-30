import os
from ML_pipeline.train_model_original import train_bert_model
from ML_pipeline.train_model_redes_neuronais import train_redes_neuronais
from ML_pipeline.train_model_rf import train_random_forest
from ML_pipeline.train_model_b_rf import train_random_forest_binary
from XAI.XAI_rf import explainable_rf
from XAI.XAI_rf_b import explain_binary_random_forest_with_lime
from XAI.XAI_neural_network import explainable_nn
from XAI.XAI_nn_mc import explainable_nn_multiclass
from ML_pipeline.test_model import test_model,test_binary_model

if __name__ == "__main__":
    print(f"Current Working Directory: {os.getcwd()}")

    print("Iniciando o treino do modelo...")
    #train_redes_neuronais()
    #train_random_forest()
    explainable_rf("ML_Program/saved_models/random_forest_20241130_012230/random_forest_model.pkl")
    #train_random_forest_binary()
    #test_model("ML_program/saved_models/random_forest_20241129_235943/random_forest_model.pkl")
    #test_binary_model("ML_program/saved_models/random_forest_20241129_235943/random_forest_model.pkl")
    #explain_binary_random_forest_with_lime("ML_program/saved_models/random_forest_20241129_235943/random_forest_model.pkl")
    #explainable_nn("ML_program/saved_models/model.h5")
    #explainable_nn_multiclass("ML_program/saved_models/model.h5")
    print("Pipeline concluído!")
