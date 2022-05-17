# CIG
Cell image generation


Saker att kolla på:
IMPLEMENTATION/TRÄNING/KOD
- fixa datan, maskerna och raw bilder. Nils tror jag skulle fixa så att de ser ut typ som i deras implementation (jupyter-filen ligger i facebook). Just nu inget noise, se så det blir bra med och normalisering som just nu görs i training.py (preprocces_data). Noise och resterande med bilderna sker i distort_images.py och loader.py just nu. Får vi inte till det kanske vi kan ta det från deeptrack, köra exakt de dom gör i sin notebook?

- när datan väl är ok testa köra, training.py ska funka. Troligen är loss funktionerna inte helt korrekt, speciellt delen i loss_fn_gen() där det är massa mean över olika dimensioner. Detta kommer från at outputen av discriminatorn just nu är (4x4x1), ska helst vara (8x8). 

RAPPORT
- Vi har påbörjat bakgrunden, behövs lite om vårt problem och säkert behöver det mesta utvidgas i bakggrunden.
- Metod kanske man kan skriva en del på, typ vår implementation av generator, discriminator. Att vi hämtat inspiration från deras artikel osv.
- Kanske påbörja spåna diskussionsämnen, ett är ju hur etiskt det är att utvecklas saker som skapar fake-bilder med deepfakes osv.
