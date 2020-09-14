Izvorna koda za presernovo nalogo: 

##Uporaba globokega ucenja s konvolucijskimi nevronskimi mrezami pri nuklearno-medicinski diagnostiki prekomerno delujocega tkiva obscitnic

###Navodila za uporabo:

1. Za uporabo so potrebni podatki iz studije NCT03203668 ali kaka 
druga PET/CT podatkovna baza.

2. Ce uporabljamo podatkovno bazo kot iz presernove naloge (z 
dovoljenjem predstojnih), damo podatke v ustrezne mape iz constants.py, preskoci naslednji korak

***Za novo podatkovno bazo je potrebno na novo izpolniti constants.py***

2.1. Slike naj bodo v formatu, kot jih izvozi program Horos.
Uporaba konstant je definirana v ./constants/constants.py, kjer so tudi konstante definirane.
Te se nanasajo predvsem na izgled in formatiranje podatkov, zato jih je za nove podatke treba ustrezno urediti.
Pojasnjene so v komentarjih pri datoteki constants.py

2.2. Podatki naj bodo v seznamu csv, kjer bo resnicna vrednost v pod
 indeksom "histo_lokacija".

**3. Trenutno je program nastavljen za ucenje Hibridne naloge CLoc in CPr, 
kjer so kategorije za posamezne lokacije hiperaktivnega obscitnicnega tkiva ter kategorija za primere, ki nimajo hiperaktivnega tkiva ("zdravi") 

3.1. Za posamezno nalogo je potrebno prilagoditi izhode arhitektur pri resnet10 atribut "num_classes" ter spremeniti funkcijo izgube v sigmoidno za CPr

Za vec informacij o tem pregledajte training_model.py

razred Run v training_model.py je glavni razred za eksekucijo ucenja NN. trenutno je datoteka training_model.py nastavljena tako, da se izvede navzkrizna validacija.

***za evalvacijo***
so kode v evaluate.py in frequency_analysis.py in izvlek_podatkov_iz_training_model.py

***za prikaz mask*** maske se prikazejo z multi_slice_viewer in omogocajo "scrollanje" skozi rezine - njihobv prikaz 
se izvede v models/my_models.py v razredu MyModel v funkciji forward v vrsticah z 
"multi_slice_viewer.multi_slice_viewer.seg_viewer". Zakomentirajte oz. izbrisite te vrstice, ce ne zelite prikaza.

Trenutno koda ni najbolj berljiva, ob interesu jo bolje pojasnim, za dodatne info pisite na moj email:
AVTORJEVO_IME.AVTORJEV_PRIIMEK at rocketmail.com

For more information about the code, write to me (author) at:
MY_NAME.MY_LASTNAME at rocketmail.com