# Classify goods, located on a same shelf

## Task

Basing on images and json files bboxes coordinates, mark by colors all goods, located on a same shelf.

## Solution algoritm

1. Open each files pair: image and its markup.
2. Partially clusterize markup: if some bboxes are very close to each other, join them into one (**shelf_from_img()** module.) Put new bboxes coords to dataframe **df**
3. Add new features to df: top, center and bottom coords, height, width, area and ln(area), aspect ratio of object. Some of them become useless.
4. Make several clusterings of dataframe with different parameters: first one gives draft clustering by Y axis, second give X clustering, third one provides final shelf markup.
5. Results are placed to **res_clusters** array as cluster numbers for each bbox (==shelf numbers).
6. After multiplication of boolean source bboxes by **res_clusters** array, get final markup of objects.
7. Markup is placed on source image together with shelf number. Then image being saved to **results** folder.

For script check: place it to **source_data** folder.

## Results


<img src = "https://github.com/lacmus-foundation/sharp-in/blob/master/images/59.jpg">

<img src = "https://github.com/lacmus-foundation/sharp-in/blob/master/images/75.jpg">

<img src = "https://github.com/lacmus-foundation/sharp-in/blob/master/images/109.jpg">


