# CIG
Cell image generation


Saker att kolla på:
IMPLEMENTATION/TRÄNING/KOD
- fixa datan, maskerna och raw bilder. Nils tror jag skulle fixa så att de ser ut typ som i deras implementation (jupyter-filen ligger i facebook). Just nu inget noise, se så det blir bra med och normalisering som just nu görs i training.py (preprocces_data). Noise och resterande med bilderna sker i distort_images.py och loader.py just nu. Får vi inte till det kanske vi kan ta det från deeptrack, köra exakt de dom gör i sin notebook?

- när datan väl är ok testa köra, training.py ska funka. <!--- Troligen är loss funktionerna inte helt korrekt, speciellt delen i loss_fn_gen() där det är massa mean över olika dimensioner. (Detta kom från at outputen av discriminatorn varr (4x4x1), ska helst vara (8x8)). --->
- Loss-funktionerna kan vara lösta. Generator har nu $\gamma \cdot MAE$ som loss, average över batches om jag fattat det rätt. Nu är output från discriminator (8x8x1) också! 10 antar jag är batches, så körde mean över dem. 
- När jag frågade Jesus om hur man bör göra med mse för discriminator svarade han:
      "Pointwise, 

      The size of the discriminator for both fake and real images should be (in this case) 1 x 4 x 4 x 1. Then compute the mean squared error as:

      MSE = 1/n sum (fake - real) ** 2

      or simply, compile the discriminator as:

      discriminator.compile(loss = "mse")

      and then inside "train_step" compute the loss for the discriminator as:

      loss = discriminator.compiled_loss(fake, real)"

RAPPORT
- Vi har påbörjat bakgrunden, behövs lite om vårt problem och säkert behöver det mesta utvidgas i bakggrunden.
- Metod kanske man kan skriva en del på, typ vår implementation av generator, discriminator. Att vi hämtat inspiration från deras artikel osv.
- Kanske påbörja spåna diskussionsämnen, ett är ju hur etiskt det är att utvecklas saker som skapar fake-bilder med deepfakes osv.
  - Ide för diskussion/test: Skillnad mellan att köra "label"-bilderna, asså 0-255 och bara tre kategorier
