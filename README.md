# DeepTetrad Package
DeepTetrad: high-throughput analysis of meiotic tetrads by deep learning

### Acknowledgement
- The project was initiated by Prof. Choi and the code is written by Dr. Lim (abysslover) at Pohang Science and Technology (POSTECH) Plant Genomic Recombination ([PGR](https://hae02032008.wixsite.com/kyuha)) Laboratory.
    
- This research was conducted in collaborations with the following people:
Eun-Cheon Lim<sup>1</sup>, Jaeil Kim<sup>1</sup>, Jihye Park<sup>1</sup>, Eun-Jung Kim<sup>1</sup>, Yeong Mi Park<sup>1</sup>, Juhyun Kim<sup>1</sup>, Hyun Seob Cho<sup>1</sup>, Ian R Henderson<sup>2</sup>, Gregory P Copenhaver<sup>3</sup>, Ildoo Hwang<sup>1</sup> and Kyuha Choi<sup>1</sup>.


1. Department of Life Sciences, Pohang University of Science and Technology, Pohang, Gyeongbuk, Republic of Korea
2. Department of Plant Sciences, University of Cambridge, Cambridge CB2 3EA, UK
3. Department of Biology and the Integrative Program for Biological and Genome Sciences, University of North Carolina at Chapel Hill, Chapel Hill, North Carolina, USA


## Getting Started
A step-by-step tutorial is provided in the following sections.

### Prerequisites
You should install CUDA-enabled GPU cards with at least 12GB GPU memory manufactured by nVidia, e.g., Titan XP.

### Prepare fluorescent images

1. Take photos of pollen tetrads by using microscopy.
2. The resolution of a fluorescent image must be **(2560x1920)**.
3. The **SUFFIX** of filenames is **VERY IMPORTANT** to be recognized by DeepTetrad. The conventions are listed below:

<p align="center">

|  Filename        |  Description  |
|------------ | -------------
|  I3bc_(1)**-1**.jpg      |  a **bright-field** image (first)  |
|  I3bc_(1)**-2**.jpg      |  a fluorescent image of **RED** light wave (second)  |
|  I3bc_(1)**-3**.jpg |  a fluorescent image of **GREEN** light wave (third)  |
|  I3bc_(1)**-4**.jpg |  a fluorescent image of **BLUE** light wave (fourth)  |   

</p>
<p align="center"><img width="128" height="128" src="https://github.com/abysslover/deeptetrad/raw/master/assets/I3bc_(1)-1.jpg"><img width="128" height="128" src="https://github.com/abysslover/deeptetrad/raw/master/assets/I3bc_(1)-2.jpg"><img width="128" height="128" src="https://github.com/abysslover/deeptetrad/raw/master/assets/I3bc_(1)-3.jpg"><img width="128" height="128" src="https://github.com/abysslover/deeptetrad/raw/master/assets/I3bc_(1)-4.jpg"></p>

### Prepare folder structures
1. Put fluorescent images into folders.
    - The parent folder name determines how DeepTetrad recognizes the name of samples. For instance, the parent folder name of I3bc (1) represents the sample name of fluorescent images.
    - The folder name will be matched against the names in the physical T-DNA locations when determining Tetrad types. For example, "I3bc" will be matched with "Cyan-Yellow-Red", which is the order of the protein colors of the I3bc Fluorescent Tagged Transgenic Line(FTL).

<p align="center"><img width="460" height="300" src="https://github.com/abysslover/deeptetrad/raw/master/assets/folder_structure.jpg"></p>

2. Prepare a T-DNA map file
    - The file is a plain text file in which FTL names with T-DNA color orders are listed.
    - An example of "T-DNA.txt" is shown below:
    
```
I1bc	GRC
I1fg	GCR
I2ab	CGR
I2fg	RGC
I3bc	CGR
I5ab	RGC
```

3. Examples
     - A folder structure for testing
<p align="center"><img width="460" height="300" src="https://github.com/abysslover/deeptetrad/raw/master/assets/test_folder_files_00.jpg"></p>
     - Fluorescent images for testing
<p align="center"><img width="460" height="300" src="https://github.com/abysslover/deeptetrad/raw/master/assets/test_folder_files_01.jpg"></p>

### Install DeepTetrad
1. Install Anaconda
   - Download an Anaconda distribution: [Link](https://www.anaconda.com/distribution/)
2. Create a Conda environment
```
	conda create -n pac
```
3. Install DeepTetrad
```
	conda activate pac
	conda install -c abysslover deeptetrad
```
4. Run DeepTetrad
```
	deeptetrad --physical_loc=T_DNA.txt --path=./test
```
**NOTE**: you must activate the conda enviroment before running deeptetrad if you opened a **new console**.
<p align="center"><img width="460" height="300" src="https://github.com/abysslover/deeptetrad/raw/master/assets/run_deeptetrad.jpg"></p>

### Contact
Please contact abysslover@gmail.com and kyuha@postech.ac.kr if you have any questions about DeepTetrad.
