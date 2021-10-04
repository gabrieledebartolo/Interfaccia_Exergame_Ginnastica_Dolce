# README.md
Progettazione e Sviluppo dell’Interfaccia Utente di un Exergame per la Ginnastica Dolce


Per il codice originale consultare il seguente link gitHub: https://github.com/eifrank/cv-fitness.

Questo progetto è testato su Windows 10.


Il sistema è in grado di riconoscere sei tipi di azioni: ['shoulders', 'foldedLegs', 'legs', 'jumpingJack', 'arms','squat'].


I 5 script principali sono sotto src, sono denominati in base all'ordine di esecuzione:

src/s1_get_skeletons_from_training_imgs.py \   
src/s2_put_skeleton_txts_to_a_single_txt.py  \
src/s3_preprocess_features.py \
src/s4_train.py \
src/s5_interface.py \

I primi quattro script sono presenti all'interno del repository sopra citato. 
Lo script src / s5_interface.py serve per far eseguire lo script dell'exergame, oltre che al riconoscimento delle azioni in tempo reale.

<br></br>
**Come eseguire lo script** 


*Prova su file video*: \
python src/s5_interface.py \
    --model_path model/trained_classifier.pickle \
    --data_type video \
    --data_path data_test/exercise.avi \
    --output_folder output  

*Link al dataset personalizzato*:  https://drive.google.com/file/d/1g6jBw2GowwCSWbWct0YbORafmxVvfC01/view?usp=sharing

Prima di eseguire il file *s5_interface.py* decomprimere la cartella *src/images*, fondamentale per il funzionamento del sistema.
Inoltre, è stato modificato il file *config/config.yaml* per permettere l'archivio e il caricamento dei dati dell'utente ed è stato inserito un nuovo file *config/badges.yaml* per tenere traccia dei risultati ottenuti dall'utente nell'utilizzo del sistema (punti esperienza e badge sbloccati).

 
