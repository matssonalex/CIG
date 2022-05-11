from PIL import Image

for i in range(20):
    if i < 10:
        nr = '0' + f'{i}'
    else:
        nr = f'{i}'
    # Change the path to your own
    labels_import = '/Users/Nisse/Documents/Chalmers/MPCAS/AdvNN/CIG/groundtruth-drosophila-vnc/stack1/labels/' + 'labels000000' + nr + '.png'
    membranes_import = '/Users/Nisse/Documents/Chalmers/MPCAS/AdvNN/CIG/groundtruth-drosophila-vnc/stack1/membranes/' + nr + '.png'
    mitochondria_import = '/Users/Nisse/Documents/Chalmers/MPCAS/AdvNN/CIG/groundtruth-drosophila-vnc/stack1/mitochondria/' + nr + '.png'
    raw_import = '/Users/Nisse/Documents/Chalmers/MPCAS/AdvNN/CIG/groundtruth-drosophila-vnc/stack1/raw/' + nr + '.tif'
    synapses_import = '/Users/Nisse/Documents/Chalmers/MPCAS/AdvNN/CIG/groundtruth-drosophila-vnc/stack1/synapses/' + nr + '.png'

    im_labels = Image.open(f'{labels_import}')
    im_membranes = Image.open(f'{membranes_import}')
    im_mitochondria = Image.open(f'{mitochondria_import}')
    im_raw = Image.open(f'{raw_import}')
    im_synapses = Image.open(f'{synapses_import}')


    width, height = im_labels.size
    counter = 0
    for crop_vertical in range(4):
        top = 0 + (height / 4) * crop_vertical
        bottom = (height/4) * (crop_vertical + 1)
        for crop_horizontal in range(4):
            left = (width / 4) * crop_horizontal
            right = (width/4) * (crop_horizontal + 1)
            labels_cropped = im_labels.crop((left, top, right, bottom))
            membranes_cropped = im_membranes.crop((left, top, right, bottom))
            mitochondria_cropped = im_mitochondria.crop((left, top, right, bottom))
            raw_cropped = im_raw.crop((left, top, right, bottom))
            synapses_cropped = im_synapses.crop((left, top, right, bottom))

            labels_txt = 'cropped_labels/' + 'im_'+ nr + '_' + f'{counter}' +'.png'
            membranes_txt = 'cropped_membranes/' + 'im_'+ nr + '_' + f'{counter}' +'.png'
            mitochondria_txt = 'cropped_mitochondria/' + 'im_'+ nr + '_' + f'{counter}' +'.png'
            raw_txt = 'cropped_raw/' + 'im_'+ nr + '_' + f'{counter}' +'.tif'
            synapses_txt = 'cropped_synapses/' + 'im_'+ nr + '_' + f'{counter}' +'.png'

            labels_cropped.save(labels_txt)
            membranes_cropped.save(membranes_txt)
            mitochondria_cropped.save(mitochondria_txt)
            raw_cropped.save(raw_txt)
            synapses_cropped.save(synapses_txt)
            counter += 1
            