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

* 0 <= s.length <= 3 * 104
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