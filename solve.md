[toc]

# Leetcode 刷题记录

## 986. 区间列表的交集
### 题目原文
给定两个由一些 闭区间 组成的列表，firstList 和 secondList ，其中 firstList[i] = [starti, endi] 而 secondList[j] = [startj, endj] 。每个区间列表都是成对 不相交 的，并且 已经排序 。

返回这两个区间列表的交集 。

形式上，闭区间 [a, b]（其中 a <= b）表示实数 x 的集合，而 a <= x <= b 。

两个闭区间的 交集 是一组实数，要么为空集，要么为闭区间。例如，[1, 3] 和 [2, 4] 的交集为 [2, 3] 。

**示例 1：**
```
输入：firstList = [[0,2],[5,10],[13,23],[24,25]], secondList = [[1,5],[8,12],[15,24],[25,26]]
输出：[[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
```

**示例 2：**
```
输入：firstList = [[1,3],[5,9]], secondList = []
输出：[]
```

**示例 3：**
```
输入：firstList = [], secondList = [[4,8],[10,12]]
输出：[]
```

**示例 4：**
```
输入：firstList = [[1,7]], secondList = [[3,10]]
输出：[[3,7]]
```


**提示：**

* 0 <= firstList.length, secondList.length <= 1000
* firstList.length + secondList.length >= 1
* 0 <= starti < endi <= 109
* endi < starti+1
* 0 <= startj < endj <= 109
* endj < startj+1

### 我的解

对于一个区间 f，在 secondList 里分别找两个区间，分别是左端点小于等于 f 左右端点的最大区间。

代码：

```Python
def search_left(interval : list, target : int) -> int:
    '''
    返回小于等于 target 的最大下标
    '''
    left = 0
    right = len(interval)

    while left < right:
        mid = (left + right) // 2

        if interval[mid][0] <= target:
            left = mid + 1
        else:
            right = mid

    return left - 1


class Solution:
    def intervalIntersection(self, firstList: list, secondList: list) -> list:
        out = []
        for single_list in firstList:
            a, b = single_list
            
            left_second_index = search_left(secondList, a)
            right_second_index = search_left(secondList, b)

            if left_second_index != -1:
                a_s, b_s = secondList[left_second_index]
                if a <= b_s:
                    out.append([max(a, a_s), min(b, b_s)])
            
            if right_second_index != -1 and 
            	right_second_index != left_second_index:
                out += secondList[
                    left_second_index + 1 : 
                    right_second_index
                ]   # 夹在找到的两个区间里面的区间，把它们合并到结果里

                a_s, b_s = secondList[right_second_index]
                if a_s <= b:
                    out.append([max(a, a_s), min(b, b_s)])
                
        return out
```

时间复杂度 $ O(n \log n)$ .

### 最优解

最优解是双指针，气坏了。

**思路**

我们称 b 为区间 [a, b] 的末端点。

在两个数组给定的所有区间中，假设拥有最小末端点的区间是 ``A[0]``。（为了不失一般性，该区间出现在数组 A 中)

然后，在数组 B 的区间中， ``A[0]`` 只可能与数组 B 中的至多一个区间相交。（如果 B 中存在两个区间均与 ``A[0]`` 相交，那么它们肯定有一个的末端点被包含在 ``A[0]`` 中，与 ``A[0]`` 拥有最小末端点矛盾）

**算法**

如果 `A[0]` 拥有最小的末端点，那么它只可能与 B[0] 相交。然后我们就可以删除区间 `A[0]`，因为它不能与其他任何区间再相交了。

相似的，如果 `B[0]` 拥有最小的末端点，那么它只可能与区间 `A[0]` 相交，然后我们就可以将 `B[0]` 删除，因为它无法再与其他区间相交了。

我们用两个指针 `i` 与 `j` 来模拟完成删除 `A[0]` 或 `B[0]` 的操作。

```python
class Solution:
    def intervalIntersection(self, firstList: list, secondList: list) -> list:
        out = []
        length_A = len(firstList)
        length_B = len(secondList)

        i = j = 0

        while i < length_A and j < length_B:
            lo = max(firstList[i][0], secondList[j][0])
            hi = min(firstList[i][1], secondList[j][1])

            if lo <= hi:
                out.append([lo, hi])

            if firstList[i][1] < secondList[j][1]:
                i += 1
            else:
                j += 1
                
        return out
```

时间复杂度 $O(n + m)$.

## 89. 格雷编码
### 题目原文

**n 位格雷码序列** 是一个由 $2^n$ 个整数组成的序列，其中：
* 每个整数都在范围 [0, $2^n$ - 1] 内（含 0 和 $2^n$ - 1）
* 第一个整数是 0
* 一个整数在序列中出现 **不超过一次**
* 每对 相邻 整数的二进制表示 **恰好一位不同** ，且
* **第一个** 和 **最后一个** 整数的二进制表示 **恰好一位不同**

给你一个整数 n ，返回任一有效的 n 位格雷码序列 。

### 我的解

格雷码数学原理 $g(i) = b(i+1) \oplus b(i),~~~~0 \le i \lt n$

其中 $\oplus$ 是按位异或运算，$b(i), g(i)$ 分别是二进制码和格雷码的第 $i$ 位，且 $b(n) = 0$

代码：

```c
int* grayCode(int n, int* returnSize) 
{
    int ret_size = 1 << n;
    int* ret = (int*)malloc(ret_size * sizeof(int));
    
    for (int i = 0; i < ret_size; i++) 
    {
        ret[i] = (i >> 1) ^ i;
    }
    *returnSize = ret_size;
    return ret;
}
```