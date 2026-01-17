import numpy as np
# def iterate_n_elements(lst, n):
#     for i in range(0, len(lst), n):
#         yield lst[i:i+n]
 
# # 示例
# lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# for group in iterate_n_elements(lst, 3):
#     print(group)

# lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# for group in iterate_n_elements(lst, 3):
#     print(group)



# lst1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# lst2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# lst = [lst1, lst2]
# print(*lst)
# for index, n in enumerate(zip(*lst)):
#     print(len(n))
#     print(index, n)

bboxes_list = []
bboxes_list1 = []
line = 'bird_94_000114.jpg 802,164,831,179,0,0.375 912,352,933,369,0,0.375'
line =  line.split()
bboxes = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])
bboxes_list1.append(bboxes)
bboxes_list += bboxes_list1

bboxes_list2 = []
line = 'bird_65_000103.jpg 111,470,187,504,0,0.875 965,121,1046,162,0,0.875'
line =  line.split()
bboxes = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])
bboxes_list2.append(bboxes)
bboxes_list += bboxes_list2

print(bboxes_list)
bboxes = np.concatenate(bboxes_list, axis=0)
print(bboxes)
print(bboxes.shape)
print(type(bboxes))