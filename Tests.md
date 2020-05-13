# Test table

| | CUDA | ScatterAlloc | Halloc | XMalloc | Ouroboros | Reg-Eff | FDGMalloc | BulkAlloc|
|:---:|:---:|:---:| :---:|:---:|:---:|:---:|:---:|:---:|
|Performance 10K|:heavy_check_mark:|-|-|-|-|-|-|-|
|Performance 100K|-|-|-|-|-|-|-|-|
|Mixed 10K|:heavy_check_mark:|:heavy_check_mark:|-|:heavy_check_mark:|-|-|-|-|
|Mixed 100K|:heavy_check_mark:|:heavy_check_mark:|-|:interrobang:|-|-|-|-|
|Scaling 2¹ - 2²⁰|:heavy_check_mark:|:heavy_check_mark:|-|-|-|-|-|-|
|Fragmentation 1|-|-|-|-|-|-|-|-|
|Fragmentation 2|-|-|-|-|-|-|-|-|
|Graph Initialization|-|-|-|-|-|-|-|-|
|Graph Updates|-|-|-|-|-|-|-|-|

## Notes Performance
* Text

## Notes Scaling
* Text

## Notes Mixed
* `XMalloc`
  *  for `100.000` allocations with range `512-8192`
  * `10.000` works without a problem

## Notes Fragmentation
* Text

## Notes Dynamic Graph
* Text