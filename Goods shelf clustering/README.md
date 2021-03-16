# Classify goods, located on a same shelf

## Task

Basing on images and json files with bboxes coordinates, mark by colors all goods, located on a same shelf.

## Solution algoritm

1. Open each files pair: image and its markup.
2. Partially clusterize markup: if some bboxes are very close to each other, join them into one (**shelf_from_img()** module.) Put new bboxes coords to dataframe **df**.
3. Add new features to **df**: top, center and bottom coords, height, width, area and ln(area), aspect ratio of object. Some of them become useless.
4. Make several clusterings of dataframe with different parameters: first one gives draft clustering by Y axis, second give X clustering, third one provides final shelf markup.
5. Results are placed to **res_clusters** array as cluster numbers for each bbox (==shelf numbers).
6. After multiplication of boolean source bboxes by **res_clusters** array, get final markup of objects.
7. Markup is placed on source image together with shelf number. Then image being saved to **results** folder.

For script check: place it to **source_data** folder.

## Results


<img src = "https://github.com/2326wz/Test_tasks/tree/master/Goods%20shelf%20clustering/results/shelf_clustered_2df842e3-2a23-46f0-8ef5-4e4806cfb92e_0.jpeg", width=300>

<img src = "https://github.com/2326wz/Test_tasks/tree/master/Goods%20shelf%20clustering/results/shelf_clustered_24d8bfc3-af64-4525-9dfc-bab6d1df6cbe_0.jpeg", width=300>

<img src = "https://github.com/2326wz/Test_tasks/tree/master/Goods%20shelf%20clustering/results/shelf_clustered_47532b0d-96bc-4589-983f-1e3e5700626a_1.jpeg", width=300>

<img src = "https://github.com/2326wz/Test_tasks/tree/master/Goods%20shelf%20clustering/results/shelf_clustered_d3518d48-72d7-47f8-b334-4e61244c4378_0.jpeg", width=300>

<img src = "https://github.com/2326wz/Test_tasks/tree/master/Goods%20shelf%20clustering/results/shelf_clustered_f3629039-af8c-4072-b956-32b77b762f0d_4.jpeg", width=300>

![](/results/shelf_clustered_2df842e3-2a23-46f0-8ef5-4e4806cfb92e_0.jpeg)


