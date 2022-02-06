# deepCOLOR: DEEP Generative model for single-cell COLOcalization Representation
DeepCOLOR is intended to analyze colocalization relation ships between single cell transcriptomes, integrating them with spatial transcriptome.

## Instalation
You can install deepCOLOR using pip command from your shell.
```shell
pip install deepCOLOR
```

## Usage
You need to prepare [`AnnData` objects](https://anndata.readthedocs.io/en/latest/) which includes raw count matrix of gene expression for single cell and spatial transcriptome respectively. You can see the usage in [IPython Notebook](tutorial/deepcolor_tutorial.ipynb).