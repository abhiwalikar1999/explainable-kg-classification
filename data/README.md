# Data

`aifb-hetero/` holds the AIFB dataset, a knowledge graph about the people,
publications, projects and research groups of the AIFB institute (KIT).

| File | Content |
|---|---|
| `aifbfixed_complete.n3` | The RDF graph (N3, 29,226 triples before leakage removal) |
| `trainingSet.tsv` / `testSet.tsv` | Standard 140 / 36 person split with research-group labels |
| `completeDataset.tsv` | All 176 labelled persons |

Source: [AIFB DataSet on figshare](https://figshare.com/articles/dataset/AIFB_DataSet/745364/1).
Original dataset: S. Bloehdorn and Y. Sure, *Kernel Methods for Mining
Instance Data in Ontologies*, ISWC 2007.

The label relation (`swrc:affiliation`), its inverse (`swrc:employs`) and the
research groups' `member` / `head` edges are removed when the graph is loaded
(`src/dataset.py`). They stay in the raw file so the removal is visible and
tested.
