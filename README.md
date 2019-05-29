# DeepTetrad Package

A deep learning model for high-throughput fluorescent image analysis of pollen tetrads

## Getting started

1. Prepare fluorescent images
   1. Take the photos of pollen tetrads by microscopy.
   2. The suffix of filenames is very **IMPORTANT** to be recognized by DeepTetrad. The conventions are listed below:
   | Filename        | description  |
   | ------------- |:-------------:|
   | I3bc_(1)**-1**.jpg      | a **bright-field** image (first) |
   | I3bc_(1)**-2**.jpg      | a fluorescent image of **RED** light wave (second)      |
   | I3bc_(1)**-3**.jpg | a fluorescent image of **GREEN** light wave (third)      |
   | I3bc_(1)**-4**.jpg | a fluorescent image of **BLUE** light wave (fourth)      |
         <p align="center"><img width="128" height="128" src="https://github.com/abysslover/deeptetrad/raw/master/assets/I3bc_(1)-1.jpg"><img width="128" height="128" src="https://github.com/abysslover/deeptetrad/raw/master/assets/I3bc_(1)-2.jpg"><img width="128" height="128" src="https://github.com/abysslover/deeptetrad/raw/master/assets/I3bc_(1)-3.jpg"><img width="128" height="128" src="https://github.com/abysslover/deeptetrad/raw/master/assets/I3bc_(1)-4.jpg"></p>
   3. Put fluorescent images into folders.
      1. The parent folder name determines how DeepTetrad recognizes the name of samples. For instance, the parent folder name of I3bc (1) represents the sample name of fluorescent images. This folder name will be matched against the names in the physical T-DNA locations when determining Tetrad types. For example, "I3bc" will be matched with "Cyan-Yellow-Red", which is the order of the protein colors of the I3bc Fluorescent Tagged Transgenic Line(FTL).
    <p align="center"><img width="460" height="300" src="https://github.com/abysslover/deeptetrad/raw/master/assets/folder_structure.jpg"></p>