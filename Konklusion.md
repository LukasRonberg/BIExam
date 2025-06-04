
## Konklusion på vores eksamensprojekt

Gennem vores projekt har vi arbejdet med et følsomt og vigtigt emne – selvmordsstatistik fordelt på lande og aldersgrupper i verden – og forsøgt at analysere og forstå mønstre og potentielle risikofaktorer ved hjælp af Business Intelligence-værktøjer som dataindsamling, datarensning, statistisk analyse og visualisering, kombineret med machine learning-metoder. Vi har anvendt både supervised og unsupervised modeller, herunder Decision Trees, Random Forest, KNN og K-means clustering, til at undersøge datasættet og afprøve muligheden for at lave forudsigelser.

Et centralt valg i vores proces var beslutningen om at arbejde med en version af datasættet, hvor outliers var fjernet. Dette blev gjort for at forbedre modellernes performance og sikre, at resultaterne ikke blev domineret af ekstreme værdier.

Det viste sig især at være gavnligt for modeller som K-Nearest Neighbors (KNN), Decision Tree og Random Forest. I Random Forest og Decision Trees forbedrede fjernelsen af outliers modellernes evne til at generalisere og træffe mere stabile beslutninger. For KNN var det særligt vigtigt, fordi modellen baserer sig på afstande mellem observationer, og outliers kan derfor forvrænge nærmeste naboers betydning. Det rensede datasæt skabte dermed en mere stabil og struktureret basis for modellering og gjorde det muligt at bygge mere robuste og troværdige modeller.

Dog er det vigtigt at understrege, at man i arbejdet med sundhedsdata – som vores – også kan argumentere for at bevare outliers. Ekstreme tilfælde kan netop indeholde værdifuld information, da de ofte repræsenterer reelle, men sjældne hændelser i befolkningen. Derfor kan det oprindelige datasæt, inklusive outliers, i visse sammenhænge give et mere realistisk billede af virkeligheden og bør ikke nødvendigvis sorteres fra, især hvis formålet er at forstå og reagere på alvorlige enkelttilfælde som selvmord.

På trods af vores grundige datarensning og modellering må vi dog konstatere, at datasættet i sin nuværende form er for mangelfuldt til at kunne bruges til sikre og troværdige forudsigelser. Flere centrale oplysninger – som f.eks. sygdomstilstand, psykologisk vurdering, sociale forhold og adgang til hjælp – mangler. Disse oplysninger mener vi spiller en central rolle i forståelsen af selvmord, og dette gør det svært at opbygge modeller, der virkelig kan forklare eller forudsige noget brugbart. Resultaterne bør derfor tolkes med stor forsigtighed og ses som overordnede tendenser frem for klare konklusioner.

Afslutningsvis har projektet givet os værdifuld erfaring med BI-processer, dataforberedelse, modellering og kritisk vurdering af datakvalitet og modellernes anvendelighed. Det har også tydeliggjort vigtigheden af at forstå den kontekst, man arbejder i – især når man håndterer følsomme sundhedsdata. Vores løsning viser potentiale, men fremhæver også, hvor vigtigt det er at have adgang til mere komplette og centrale data/oplysninger, hvis man vil kunne lave forudsigelser, der kan gøre en reel forskel i praksis.

Vores analyser og resultater har vi gjort tilgængelige i en Streamlit-applikation, hvor brugeren kan udforske datavisualiseringer og de modeller, vi har udviklet.


