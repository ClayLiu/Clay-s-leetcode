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
        ret[i] = (i >> 1) ^ i;
    
    *returnSize = ret_size;
    return ret;
}
```

## 23. 合并K个升序链表
### 题目原文
难度：<font color = 'red'>困难</font>

给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。

**示例 1：**
```
输入：lists = [[1,4,5],[1,3,4],[2,6]]
输出：[1,1,2,3,4,4,5,6]
解释：链表数组如下：
[
  1->4->5,
  1->3->4,
  2->6
]
将它们合并到一个有序链表中得到。
1->1->2->3->4->4->5->6
```

**示例 2：**
```
输入：lists = []
输出：[]
```

**示例 3：**
```
输入：lists = [[]]
输出：[]
```

**提示：**

* k == lists.length
* 0 <= k <= 10^4
* 0 <= lists[i].length <= 500
* -10^4 <= lists[i][j] <= 10^4
* lists[i] 按 升序 排列
* lists[i].length 的总和不超过 10^4

### 我的解

```c
#define NULL 0

struct ListNode 
{
    int val;
    struct ListNode *next;
};

void insert(struct ListNode* head, struct ListNode* inserting_node)
{
    struct ListNode* prev = head;
    while(head != NULL && head->val <= inserting_node->val)
    {
        prev = head;
        head = head->next;
    }
    
    inserting_node->next = head;
    prev->next = inserting_node;
}


struct ListNode* mergeKLists(struct ListNode** lists, int listsSize)
{
    if(listsSize == 0)
        return NULL;

    int i;
    struct ListNode* min_node = NULL;
    struct ListNode* curr_node = NULL;
    struct ListNode* next_save = NULL;


    for(i = 0; i < listsSize && (min_node = lists[i]) == NULL; i++);

    for(; i < listsSize; i++)
        if(lists[i] != NULL && min_node->val > lists[i]->val)
            min_node = lists[i];
    
    for(i = 0; i < listsSize; i++)
        if(lists[i] != NULL && lists[i] != min_node)
        {
            curr_node = lists[i];
            while(curr_node != NULL)
            {
                next_save = curr_node->next;
                insert(min_node, curr_node);
                curr_node = next_save;
            }
        }

    return min_node;
}
```

## 32.最长有效括号
难度:<font color = 'red'> 困难 </font>

### 题目原文

给你一个只包含 `'('` 和 `')'` 的字符串，找出最长有效（格式正确且连续）括号子串的长度。

**示例 1：**
```
输入：s = "(()"
输出：2
解释：最长有效括号子串是 "()"
```
**示例 2：**
```
输入：s = ")()())"
输出：4
解释：最长有效括号子串是 "()()"
```
**示例 3：**
```
输入：s = ""
输出：0
```

**提示：**

* 0 <= s.length <= 3 * $10^4$
* s[i] 为 '(' 或 ')'

### 官方解

用了栈，用栈来保存 `栈底元素为当前已经遍历过的元素中「最后一个没有被匹配的右括号的下标」`  而 `栈里其他元素维护左括号的下标` 。

需要注意的是，如果一开始栈为空，第一个字符为左括号的时候我们会将其放入栈中，这样就不满足提及的「最后一个没有被匹配的右括号的下标」，为了保持统一，我们在一开始的时候往栈中放入一个值为 −1 的元素。

* 对于遇到的每个 `(`，我们将它的下标放入栈中
* 对于遇到的每个 `)`，我们先弹出栈顶元素表示匹配了当前右括号:
    * 如果栈为空，说明当前的右括号为没有被匹配的右括号，我们将其下标放入栈中来更新我们之前提到的「最后一个没有被匹配的右括号的下标」
    * 如果栈不为空，当前右括号的下标减去栈顶元素即为「以该右括号为结尾的最长有效括号的长度」

代码：

```python
class Solution:
    def longestValidParentheses(self, s : str) -> int:
        stack = [-1]
        max_length = 0

        for i, c in enumerate(s):
            if c == '(':
                stack.append(i)
            else:
                stack.pop()
                if stack:
                    max_length = max(max_length, i - stack[-1])
                else:
                    stack.append(i)

        return max_length
```

时间复杂度 $O(n)$，空间复杂度 $O(n)$.

优化：

在此方法中，我们利用两个计数器 `left` 和 `right` 。首先，我们从左到右遍历字符串，对于遇到的每个 `'('`，我们增加 `left` 计数器，对于遇到的每个 `')'` ，我们增加 `right` 计数器。每当 `left` 计数器与 right 计数器相等时，我们计算当前有效字符串的长度，并且记录目前为止找到的最长子字符串。当 `right` 计数器比 `left` 计数器大时，我们将 `left` 和 `right` 计数器同时变回 0。

这样的做法贪心地考虑了以当前字符下标结尾的有效括号长度，每次当右括号数量多于左括号数量的时候之前的字符我们都扔掉不再考虑，重新从下一个字符开始计算，但这样会漏掉一种情况，就是遍历的时候左括号的数量始终大于右括号的数量，即 `(()` ，这种时候最长有效括号是求不出来的。

解决办法，反过来再来一遍。

```c
int longestValidParentheses(char * s)
{
    int i;
    int left, right, max_length = 0;
    left = right = 0;

    for(i = 0; s[i]; i++)
    {
        if(s[i] == '(')
            left++;
        else
            right++;
   
        if(left == right)
        {
            if(max_length < left)
                max_length = left;
        }
        else if(right > left)
            left = right = 0;
    }

    left = right = 0;
    while(i)
    {
        i--;
        if(s[i] == '(')
            left++;
        else
            right++;
        
        if(left == right)
        {
            if(max_length < left)
                max_length = left;
        }
        else if(right < left)
            left = right = 0;
    }
    
    return max_length * 2;
}
```
时间复杂度 $O(n)$，空间复杂度 $O(1)$.

## 306. 累加数
### 题目原文
难度：<font color = 'orange'>中等</font>

**累加数** 是一个字符串，组成它的数字可以形成累加序列。

一个有效的 **累加序列** 必须 **至少** 包含 3 个数。除了最开始的两个数以外，字符串中的其他数都等于它之前两个数相加的和。

给你一个只包含数字 `'0'-'9'` 的字符串，编写一个算法来判断给定输入是否是 累加数 。如果是，返回 `true` ；否则，返回 `false` 。

**说明**：累加序列里的数 **不会** 以 `0` 开头，所以不会出现 `1`, `2`, `03` 或者 `1`, `02`, `3` 的情况。

**示例 1：**
```
输入："112358"
输出：true 
解释：累加序列为: 1, 1, 2, 3, 5, 8 。1 + 1 = 2, 1 + 2 = 3, 2 + 3 = 5, 3 + 5 = 8
```
**示例 2：**
```
输入："199100199"
输出：true 
解释：累加序列为: 1, 99, 100, 199。1 + 99 = 100, 99 + 100 = 199
```

**提示：**

* 1 <= num.length <= 35
* num 仅由数字（0 - 9）组成

### 我的解（看了官方解的提示后写的）

累加序列，由开始两个数字确定。而数字的长度不确定，因此要枚举前两个数字的组合。

记第一个数字的下标是 [0 : first_end]，第二个数字的下标是 [second_start : second_end]，由于 second_start = first_end + 1，因此只要枚举 second_start 和 second_end 和组合即可。

然后因为数字可能超过 `unsigend long long` 的大小，故要一个高精度大数运算的模块，（果断用 Python）

综上得代码：

```python
class Solution:
    def isAdditiveNumber(self, num : str) -> bool:
        length = len(num)
        for second_start in range(1, length - 1):
            for second_end in range(second_start + 1, length):
                first = int(num[:second_start])
                second = int(num[second_start : second_end])

                if self.if_valid(first, second, num):
                    return True
        
        return False


    def if_valid(self, first : int, second : int, num : str) -> bool:
        length = len(num)
        temp_str = str(first) + str(second)
        
        while len(temp_str) < length:
            thrid = first + second
            temp_str += str(thrid)

            first = second
            second = thrid

        return temp_str == num
```

时间复杂度 $O(n^3)$，空间复杂度 $O(n)$.

## 334. 递增的三元子序列
### 题目原文

难度：<font color = 'orange'>中等</font>

给你一个整数数组 `nums` ，判断这个数组中是否存在长度为 3 的递增子序列。

如果存在这样的三元组下标 `(i, j, k)` 且满足 `i < j < k` ，使得 `nums[i] < nums[j] < nums[k]` ，返回 `true` ；否则，返回 `false` 。

**示例 1：**
```
输入：nums = [1,2,3,4,5]
输出：true
解释：任何 i < j < k 的三元组都满足题意
```
**示例 2：**
```
输入：nums = [5,4,3,2,1]
输出：false
解释：不存在满足题意的三元组
```
**示例 3：**
```
输入：nums = [2,1,5,0,4,6]
输出：true
解释：三元组 (3, 4, 5) 满足题意，因为 nums[3] == 0 < nums[4] == 4 < nums[5] == 6
```

**提示：**

* 1 <= nums.length <= 5 * $10^5$
* -$2^{31}$ <= nums[i] <= $2^{31}$ - 1

### 我的解（看了提示之后的解）

使用两个数组 `left_min`, `right_max` 分别表示左右前缀（后缀）最小值（最大值）。

`left_min[i]` 表示在 `nums[0:i]` （左右闭区间）中的最小值。
`right_max[i]` 表示在 `nums[i:n - 1]` （左右闭区间）中的最大值。

那么如果 `left_min[i] < nums[i] < right_max[i]`，即证明存在。

综上得代码：

```c
#define min(x, y) ((x) < (y) ? (x) : (y))
#define max(x, y) ((x) > (y) ? (x) : (y))

bool increasingTriplet(int* nums, int numsSize)
{
    int i;
    bool flag = 0;
    int* left_min = malloc(sizeof(int) * numsSize);
    int* right_max = malloc(sizeof(int) * numsSize);

    left_min[0] = nums[0];
    for(i = 1; i < numsSize; i++)
        left_min[i] = min(left_min[i - 1], nums[i]);
    
    right_max[numsSize - 1] = nums[numsSize - 1];
    for(i = numsSize - 2; i >= 0; i--)
        right_max[i] = max(right_max[i + 1], nums[i]);

    for(i = 1; i < numsSize - 1; i++)
        if(left_min[i] < nums[i] && nums[i] < right_max[i])
        {
            flag = 1;
            break;
        }
    
    free(left_min);
    free(right_max);

    return flag;
}
```

时间复杂度 $O(n)$，空间复杂度 $O(n)$.

### 优化

可以采用贪心的策略。使用 `first` 保存当前碰到的最小元素，`second` 保存当前碰到的第二小元素。

代码：
```c
bool increasingTriplet(int* nums, int numsSize)
{
    int i;
    int first, second;

    first = nums[0];
    second = INT_MAX;

    for(i = 1; i < numsSize; i++)
    {
        if(nums[i] > second)        // first < second < nums[i], found it!
            return 1;
        else if(nums[i] > first)    // first < nums[i] <= second, contain first < second < anything.
            second = nums[i];
        else                        // nums[i] <= first < second, contain first < second < anything.
            first = nums[i];    
    }

    return 0;
}
```

时间复杂度 $O(n)$，空间复杂度 $O(1)$.

## 51. N 皇后
### 题目原文

难度：<font color = 'red'> 困难 </font>

**n 皇后问题**研究的是如何将 `n` 个皇后放置在 `n × n` 的棋盘上，并且使皇后彼此之间不能相互攻击。

给你一个整数 `n` ，返回所有不同的**n 皇后问题** 的解决方案。

每一种解法包含一个不同的n 皇后问题 的棋子放置方案，该方案中 `'Q'` 和 `'.'` 分别代表了皇后和空位。

**示例 1：**

![](https://assets.leetcode.com/uploads/2020/11/13/queens.jpg)
```
输入：n = 4
输出：[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
解释：如上图所示，4 皇后问题存在两个不同的解法。
```

**示例 2：**

```输入：n = 1
输出：[["Q"]]
```

**提示：**

* 1 <= n <= 9

### 我的解：

类似解数独，疯狂 dfs。

```python
class Solution:
    def valid(self, t : List[str], row : int, col : int, n : int) -> bool:    
        queen_count = 0
        for i in range(n):
            queen_count += t[i][col] == 'Q'
        
        if queen_count > 1:
            return False
        
        queen_count = 0

        for j in range(n):
            queen_count += t[row][j] == 'Q'

        if queen_count > 1:
            return False
        
        queen_count = 0

        i = 0
        while row + i < n and col + i < n:
            queen_count += t[row + i][col + i] == 'Q'
            i += 1

        i = 1
        while row - i >= 0 and col - i >= 0:
            queen_count += t[row - i][col - i] == 'Q'
            i += 1

        if queen_count > 1:
            return False
        
        queen_count = 0
        
        i = 0
        while row - i >= 0 and col + i < n:
            queen_count += t[row - i][col + i] == 'Q'
            i += 1

        i = 1
        while row + i < n and col - i >= 0:
            queen_count += t[row + i][col - i] == 'Q'
            i += 1

        if queen_count > 1:
            return False

        return True


    def solveNQueens(self, n : int) -> List[List[str]]:
        out = []
        t = [['.'] * n for _ in range(n)]

        def dfs(depth : int, queen_count : int):
            if queen_count == n:
                # print(t)
                out.append([''.join(s) for s in t])
                return

            if depth == n * n:
                return

            i = depth // n
            j = depth % n

            for place in ['Q', '.']:
                t[i][j] = place
            
                if self.valid(t, i, j, n):
                    dfs(depth + 1, queen_count + (place == 'Q'))

        dfs(0, 0)
        return out
```

很烦的是，这样做超时了。

分析：dfs 是 $O(2^{n^2})$ 的，然而判断是否有效函数 `valid`，是 $O(n)$ 的。因此整体是 $O(n\times 2^{n^2})$.

### 看了一次提示的解

看了一次官方题解，看到提示是

1. 同一主对角线的元素，`行下标` 与 `列下标` 之差一样。
2. 同一副对角线的元素，`行下标` 与 `列下标` 之和一样。
3. 同一行、同一列则很容易判断。上面做法 $O(n)$ 没有必要。

综上得代码

```python
class Solution:
    def valid(self, curr_dict : dict, i : int, j : int) -> bool:    
        rows = curr_dict['rows']
        cols = curr_dict['cols']
        diag_m = curr_dict['diag_m']
        diag_e = curr_dict['diag_e']

        if  i in rows or \
            j in cols or \
            i - j in diag_m or \
            i + j in diag_e \
        : 
            return False
        else:
            return True


    def solveNQueens(self, n : int) -> List[List[str]]:
        out = []
        t = [['.'] * n for _ in range(n)]
        curr_dict = {
            'rows' : [],
            'cols' : [],
            'diag_m' : [],
            'diag_e' : []
        }

        def dfs(depth : int, queen_count : int):
            if queen_count == n:
                # print(t)
                out.append([''.join(s) for s in t])
                return

            if depth == n * n:
                return

            i = depth // n
            j = depth % n

            t[i][j] = 'Q'
            if self.valid(curr_dict, i, j):
                curr_dict['rows'].append(i)
                curr_dict['cols'].append(j)
                curr_dict['diag_m'].append(i - j)
                curr_dict['diag_e'].append(i + j)

                dfs(depth + 1, queen_count + 1)
            
                curr_dict['rows'].pop()
                curr_dict['cols'].pop()
                curr_dict['diag_m'].pop()
                curr_dict['diag_e'].pop()

            t[i][j] = '.'
            dfs(depth + 1, queen_count)
            
        dfs(0, 0)
        return out
```

喜提通过，但是。。

![超长时间的通过](9c280d0e1bc07a9976d42d2a4df86cb.png)

分析：在上面的做法中，放置皇后是每一格都尝试放置一次。然而，放了这一行，就可以跳到下一行了。

并且，放置之后 dfs，不放置也 dfs 使得 dfs 复杂度 $O(2^{n^2})$ 。其实每一行只会有一个皇后，因此枚举每一行皇后的列坐标即可，这样复杂度就是 $O(n!)$ 了。

### 看了两次提示的解

```python
class Solution:
    def valid(self, curr_dict : dict, i : int, j : int) -> bool:    
        cols = curr_dict['cols']
        diag_m = curr_dict['diag_m']
        diag_e = curr_dict['diag_e']

        if  j in cols or \
            i - j in diag_m or \
            i + j in diag_e \
        : 
            return False
        else:
            return True


    def solveNQueens(self, n : int) -> List[List[str]]:
        out = []
        t = [['.'] * n for _ in range(n)]
        curr_dict = {
            'cols' : [],
            'diag_m' : [],
            'diag_e' : []
        }

        def dfs(row : int, queen_count : int):
            if queen_count == n:
                # print(t)
                out.append([''.join(s) for s in t])
                return

            if row == n:
                return

            for j in range(n):
                i = row
                
                if self.valid(curr_dict, i, j):
                    t[i][j] = 'Q'
                    curr_dict['cols'].append(j)
                    curr_dict['diag_m'].append(i - j)
                    curr_dict['diag_e'].append(i + j)

                    dfs(row + 1, queen_count + 1)
                
                    curr_dict['cols'].pop()
                    curr_dict['diag_m'].pop()
                    curr_dict['diag_e'].pop()

                    t[i][j] = '.'
            
        dfs(0, 0)
        return out
```

再优化一下，因为 `set()` 可以 $O(\log n)$ 地增删查，因而

```python
class Solution:
    def valid(self, curr_dict : dict, i : int, j : int) -> bool:    
        cols = curr_dict['cols']
        diag_m = curr_dict['diag_m']
        diag_e = curr_dict['diag_e']

        if  j in cols or \
            i - j in diag_m or \
            i + j in diag_e \
        : 
            return False
        else:
            return True


    def solveNQueens(self, n : int) -> List[List[str]]:
        out = []
        t = [['.'] * n for _ in range(n)]
        curr_dict = {
            'cols' : set(),
            'diag_m' : set(),
            'diag_e' : set()
        }

        def dfs(row : int, queen_count : int):
            if queen_count == n:
                # print(t)
                out.append([''.join(s) for s in t])
                return

            if row == n:
                return

            for j in range(n):
                i = row
                
                if self.valid(curr_dict, i, j):
                    t[i][j] = 'Q'
                    curr_dict['cols'].add(j)
                    curr_dict['diag_m'].add(i - j)
                    curr_dict['diag_e'].add(i + j)

                    dfs(row + 1, queen_count + 1)
                
                    curr_dict['cols'].remove(j)
                    curr_dict['diag_m'].remove(i - j)
                    curr_dict['diag_e'].remove(i + j)

                    t[i][j] = '.'
            
        dfs(0, 0)
        return out
```

时间复杂度 $O(n!)$，空间复杂度 $O(n)$.

### 官方最优解

使用位运算来判断 valid。

## 373. 查找和最小的 K 对数字
### 题目原文
难度：<font color = 'orange'> 中等 </font>

给定两个以 升序排列 的整数数组 `nums1` 和 `nums2`,以及一个整数 `k`。

定义一对值`(u,v)`，其中第一个元素来自`nums1`，第二个元素来自 `nums2`。

请找到和最小的 `k`个数对`(u1,v1)`, `(u2,v2)` ... `(uk,vk)`。



**示例 1:**
```
输入: nums1 = [1,7,11], nums2 = [2,4,6], k = 3
输出: [1,2],[1,4],[1,6]
解释: 返回序列中的前 3 对数：
     [1,2],[1,4],[1,6],[7,2],[7,4],[11,2],[7,6],[11,4],[11,6]
```
**示例 2:**
```
输入: nums1 = [1,1,2], nums2 = [1,2,3], k = 2
输出: [1,1],[1,1]
解释: 返回序列中的前 2 对数：
    [1,1],[1,1],[1,2],[2,1],[1,2],[2,2],[1,3],[1,3],[2,3]
```
**示例 3:**
```
输入: nums1 = [1,2], nums2 = [3], k = 3 
输出: [1,3],[2,3]
解释: 也可能序列中所有的数对都被返回:[1,3],[2,3]
```

**提示:**

* 1 <= nums1.length, nums2.length <= 10^5
* -10^9 <= nums1[i], nums2[i] <= 10^9
* nums1 和 nums2 均为升序排列
* 1 <= k <= 1000

### 我的解

```c
/**
 * Return an array of arrays of size *returnSize.
 * The sizes of the arrays are returned as *returnColumnSizes array.
 * Note: Both returned array and *columnSizes array must be malloced, assume caller calls free().
 */
int** kSmallestPairs(
    int* nums1, 
    int nums1Size, 
    int* nums2, 
    int nums2Size, 
    int k, 
    int* returnSize, 
    int** returnColumnSizes
)
{
    // k = nums1Size * nums2Size < k ? nums1Size * nums2Size : k;

    *returnSize = k;
    int i, j, g;
    int top = 0;
    int** out = (int**)malloc(sizeof(int*) * k);
    *returnColumnSizes = (int*)calloc(k, sizeof(int));
    int* row_access = (int*)malloc(sizeof(int) * nums1Size);

    for(i = 0; i < nums1Size * nums2Size < k ? nums1Size * nums2Size : k; i++)
    {
        out[i] = (int*)malloc(sizeof(int) * 2);
        (*returnColumnSizes)[i] = 2;
    }
    
    row_access[0] = 1;
    for(i = 1; i < nums1Size; i++)
        row_access[i] = 0;
    
    i = j = 0;
    while(k--)
    {
        out[top][0] = nums1[i];
        out[top][1] = nums2[j];
        top++;

        for(g = 0; g < nums1Size; g++)
            if(row_access[g] < nums2Size)
            {
                i = g;
                j = row_access[g];
                break;
            }
        
        if(g == nums1Size)
            break;

        for(g++; g < nums1Size; g++)
        {
            // printf("g = %d, row_access[g] = %d\n", g, row_access[g]);
            if(nums1[g] + nums2[row_access[g]] <= nums1[i] + nums2[j])
            {
                i = g;
                j = row_access[g];
            }
            else
            {
                break;
            }
        }
        row_access[i] = j + 1; 
    }
    
    free(row_access);
    return out;
}
```

思路：

用 `row_access[]` 来保存每一个 `num1` 在 `num2` 的进度。但是 leetcode 死活不通过，神了。

