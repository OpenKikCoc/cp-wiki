## 简介

链表是一种用于存储数据的数据结构，通过如链条一般的指针来连接元素。它的特点是插入与删除数据十分方便，但寻找与读取数据的表现欠佳。

## 与数组的区别

链表和数组都可用于存储数据。与链表不同，数组将所有元素按次序依次存储。不同的存储结构令它们有了不同的优势：

链表因其链状的结构，能方便地删除、插入数据，操作次数是 $O(1)$。但也因为这样，寻找、读取数据的效率不如数组高，在随机访问数据中的操作次数是 $O(n)$。

数组可以方便地寻找并读取数据，在随机访问中操作次数是 $O(1)$。但删除、插入的操作次数是 $O(n)$ 次。

## 构建链表

> [!TIP] **tip**
> 
> 构建链表时，使用指针的部分比较抽象，光靠文字描述和代码可能难以理解，建议配合作图来理解。

### 单向链表

单向链表中包含数据域和指针域，其中数据域用于存放数据，指针域用来连接当前结点和下一节点。

![](images/list.svg)

```cpp
// C++ Version
struct Node {
  int value;
  Node *next;
};
```

```python
# Python Version
class Node:
    def __init__(self, value = None, next = None): 
        self.value = value
        self.next = next
```

### 双向链表

双向链表中同样有数据域和指针域。不同之处在于，指针域有左右（或上一个、下一个）之分，用来连接上一个结点、当前结点、下一个结点。

![](images/double-list.svg)

```cpp
// C++ Version
struct Node {
  int value;
  Node *left;
  Node *right;
};
```

```python
# Python Version
class Node:
    def __init__(self, value = None, left = None, right = None): 
        self.value = value
        self.left = left
        self.right = right
```

## 向链表中插入（写入）数据

### 单向链表

流程大致如下：

1. 初始化待插入的数据 `node`；
2. 将 `node` 的 `next` 指针指向 `p` 的下一个结点；
3. 将 `p` 的 `next` 指针指向 `node`。

具体过程可参考下图：

1. ![](./images/list-insert-1.svg)
2. ![](./images/list-insert-2.svg)
3. ![](./images/list-insert-3.svg)

代码实现如下：

```cpp
// C++ Version
void insertNode(int i, Node *p) {
  Node *node = new Node;
  node->value = i;
  node->next = p->next;
  p->next = node;
}
```

```python
# Python Version
def insertNode(i, p):
    node = Node()
    node.value = i
    node.next = p.next
    p.next = node
```

### 单向循环链表

将链表的头尾连接起来，链表就变成了循环链表。由于链表首尾相连，在插入数据时需要判断原链表是否为空：为空则自身循环，不为空则正常插入数据。

大致流程如下：

1. 初始化待插入的数据 `node`；
2. 判断给定链表 `p` 是否为空；
3. 若为空，则将 `node` 的 `next` 指针和 `p` 都指向自己；
4. 否则，将 `node` 的 `next` 指针指向 `p` 的下一个结点；
5. 将 `p` 的 `next` 指针指向 `node`。

具体过程可参考下图：

1. ![](./images/list-insert-cyclic-1.svg)
2. ![](./images/list-insert-cyclic-2.svg)
3. ![](./images/list-insert-cyclic-3.svg)

代码实现如下：

```cpp
// C++ Version
void insertNode(int i, Node *p) {
  Node *node = new Node;
  node->value = i;
  node->next = NULL;
  if (p == NULL) {
    p = node;
    node->next = node;
  } else {
    node->next = p->next;
    p->next = node;
  }
}
```

```python
# Python Version
def insertNode(i, p):
    node = Node()
    node.value = i
    node.next = None
    if p == None:
        p = node
        node.next = node
    else:
        node.next = p.next
        p.next = node
```

### 双向循环链表

在向双向循环链表插入数据时，除了要判断给定链表是否为空外，还要同时修改左、右两个指针。

大致流程如下：

1. 初始化待插入的数据 `node`；
2. 判断给定链表 `p` 是否为空；
3. 若为空，则将 `node` 的 `left` 和 `right` 指针，以及 `p` 都指向自己；
4. 否则，将 `node` 的 `left` 指针指向 `p`;
5. 将 `node` 的 `right` 指针指向 `p` 的右结点；
6. 将 `p` 右结点的 `left` 指针指向 `node`；
7. 将 `p` 的 `right` 指针指向 `node`。

代码实现如下：

```cpp
// C++ Version
void insertNode(int i, Node *p) {
  Node *node = new Node;
  node->value = i;
  if (p == NULL) {
    p = node;
    node->left = node;
    node->right = node;
  } else {
    node->left = p;
    node->right = p->right;
    p->right->left = node;
    p->right = node;
  }
}
```

```python
# Python Version
def insertNode(i, p):
    node = Node()
    node.value = i
    if p == None:
        p = node
        node.left = node
        node.right = node
    else:
        node.left = p
        node.right = p.right
        p.right.left = node
        p.right = node
```

## 从链表中删除数据

### 单向（循环）链表

设待删除结点为 `p`，从链表中删除它时，将 `p` 的下一个结点 `p->next` 的值覆盖给 `p` 即可，与此同时更新 `p` 的下下个结点。

流程大致如下：

1. 将 `p` 下一个结点的值赋给 `p`，以抹掉 `p->value`；
2. 新建一个临时结点 `t` 存放 `p->next` 的地址；
3. 将 `p` 的 `next` 指针指向 `p` 的下下个结点，以抹掉 `p->next`；
4. 删除 `t`。此时虽然原结点 `p` 的地址还在使用，删除的是原结点 `p->next` 的地址，但 `p` 的数据被 `p->next` 覆盖，`p` 名存实亡。

具体过程可参考下图：

1. ![](./images/list-delete-1.svg)
2. ![](./images/list-delete-2.svg)
3. ![](./images/list-delete-3.svg)

代码实现如下：

```cpp
// C++ Version
void deleteNode(Node *p) {
  p->value = p->next->value;
  Node *t = p->next;
  p->next = p->next->next;
  delete t;
}
```

```python
# Python Version
def deleteNode(p):
    p.value = p.next.value
    p.next = p.next.next
```

### 双向循环链表

流程大致如下：

1. 将 `p` 左结点的右指针指向 `p` 的右节点；
2. 将 `p` 右结点的左指针指向 `p` 的左节点；
3. 新建一个临时结点 `t` 存放 `p` 的地址；
4. 将 `p` 的右节点地址赋给 `p`，以避免 `p` 变成悬垂指针；
5. 删除 `t`。

代码实现如下：

```cpp
void deleteNode(Node *&p) {
  p->left->right = p->right;
  p->right->left = p->left;
  Node *t = p;
  p = p->right;
  delete t;
}
```

```python
# Python Version
def deleteNode(p):
    p.left.right = p.right
    p.right.left = p.left
    p = p.right
```


## 习题

### 直观模拟

> [!NOTE] **[AcWing 826. 单链表](https://www.acwing.com/problem/content/828/)**
> 
> 题意: TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

本质相当于设置一个 dummy 节点，该节点为 0 ，真正头节点为 ne[0]

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 100010;

int e[N], ne[N], idx;

void init() {
    ne[0] = -1, idx = 1;
}

void add(int k, int x) {
    e[idx] = x, ne[idx] = ne[k], ne[k] = idx ++ ;
}

void del(int k) {
    ne[k] = ne[ne[k]];
}

int main() {
    init();
    
    int n;
    cin >> n;
    while (n -- ) {
        string op;
        int k, x;
        cin >> op;
        if (op[0] == 'H') {
            cin >> x;
            add(0, x);
        } else if (op[0] == 'I') {
            cin >> k >> x;
            add(k, x);
        } else {
            cin >> k;
            del(k);
        }
    }
    for (int i = ne[0]; ~i; i = ne[i]) cout << e[i] << ' ';
    cout << endl;
    return 0;
}
```

##### **Python**

```python
"""
1. 链表模拟的实现：
   1. 用指针模拟链表，效率低（每次都需要new一个新节点）一般在笔试题中，都不会采用这种动态的表达方式，一般只用在面试题中
   2. 用数组模拟链表，效率高。

2. 单链表：在面试题里用得最多的是邻接表；（邻接表其实是n个链表）
   邻接表最主要的应用是：存储树和图!!!（所以单链表在算法中用得最多的就是存储图和树，包括最短路问题，最小生成树，最大流问题）
   

链式前向星:用数组来模拟链表（就是静态链表）：

  1) head表示头结点（头结点最开始指向一个空节点）
  2) 用ev[i]存储某个点的值是多少，ne[i]存储某个点的next节点的下标/索引是多少（用下标关联起来的）；空节点的下标用-1来表示；idx表示下一个可以存储元素的位置索引。
  - 比如：Head->3->5->7->9->NULL (-1)
  - 表示为：ev[0]=3, ne[0]=1; ev[1]=5, ne[1]=2; ev[2]=7, ne[2]=3; ev[3]=9, ne[3]=-1。

 3) 数组模拟的链表可以做指针做的所有事情（包括排序），数组模拟的静态链表比new一个指针链表快！
 4) 头结点后添加元素：
    在e的idx处存储元素ev[idx]=x；
    该元素插入到头结点后面ne[idx]=head；
    头结点指向该元素head=idx；
    idx指向下一个可存储元素的位置idx++。
 5) 在第k个插入的数后插入一个数：
    在e的idx处存储元素ev[index]=x;
    该元素插入到第k个插入的数的后面ne[idx]=ne[k];
    第k个插入的数指向该元素ne[k]=idx;
    idx指向下一个可存储元素的位置idx++。

3. 双链表：主要用于优化某些问题
- 每个节点有两个指针，一个指向前，一个指向后；
  - 定义一个L[N], R[N] 存的左右的点是谁（位置在哪）
  - （偷个懒）直接定义下标是0的点是head，下标是1的点是tail（这两个点是两个边界，不是实际内容）
  - 这不是一个下标有序的链表

4. 邻接表
- 其实就是把每个点的所有邻边全部存储起来；是一堆单链表。
- head[i]-->o-->o-->-1 : 每一个head[i]存储的是第i个点所有的邻边，然后是用一个单链表存储起来的。
- 邻接表其实就是单链表（邻接表的代码和单链表的所有代码都是一样的）
"""



#在刷lc的listnode的题型时，还是用指针类的listnode结构来进行解答
#这一类用数组存储链表的形式主要用于后面的图和树
def insert_head(x):
    global idx, head
    ev[idx] = x
    ne[idx] = head
    head = idx
    #指针的概念 转换成 数组下标。
    idx += 1

 #删除k节点后的一个节点
def delete(k):
    ne[k] = ne[ne[k]]

 #在节点k后插入一个值为x的节点
def insert(k, x):
    global idx
    ev[idx] = x
    ne[idx] = ne[k]
    ne[k] = idx
    idx += 1

if __name__ == '__main__':

    N = 100000 
    n = int(input())

    #初始化链表
    head=-1    # head存储的是头结点的下标
    ev=[0]*N   # ev[i]表示节点i的取值
    ne=[0]*N   # ne[i]表示节点i的next指针是多少
    idx=0      # idx表示当前可以用的最新点的下标是多少 or/ idx存储当前用到了哪个点

    for i in range(n):
        op = input().split()
        if op[0] == 'H':
            insert_head(int(op[1]))
        elif op[0] == 'I':
            insert(int(op[1]) - 1, int(op[2]))
        else:
            k = int(op[1])
            if k:
                delete(k - 1)
            else:
                head = ne[head]

    p = head
    res = []
    while p != -1:
        res.append(ev[p])
        p = ne[p]

    print(' '.join(map(str, res)))
```

<!-- tabs:end -->
</details>

<br>

* * *


> [!NOTE] **[AcWing 827. 双链表](https://www.acwing.com/problem/content/829/)**
> 
> 题意: TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>
using namespace std;
const int maxn = 1e5+5;
int node[maxn], l[maxn], r[maxn], tot;

void init() {
    l[1] = 0, r[0] = 1;
    tot = 2;        // 前两个节点 0 1  真正的第一个数据从3开始存
}

void addl(int x) {
    node[++tot] = x;
    l[tot] = 0;     // 新节点左侧 是头部0
    r[tot] = r[0];  // 新节点右侧 是原头部的右侧
    l[r[0]] = tot;  // 原头部右侧的左侧 是新节点
    r[0] = tot;     // 头部右侧 是新节点
}

void addr(int x) {
    node[++tot] = x;
    r[tot] = 1;     // 新节点右侧 是尾部1
    l[tot] = l[1];  // 新节点左侧 是原尾部的左侧
    r[l[1]] = tot;  // 原尾部的左侧的右侧 是新节点
    l[1] = tot;     // 尾部左侧 是新节点
}

void addkl(int k, int x) {
    node[++tot] = x;
    r[tot] = k;     // 新节点右侧 是k
    l[tot] = l[k];  // 新节点左侧 是k左侧
    r[l[k]] = tot;  // 原k左侧的右侧 是新节点
    l[k] = tot;     // k左侧 是新节点
}

void addkr(int k, int x) {
    node[++tot] = x;
    l[tot] = k;     // 新节点左侧 是k
    r[tot] = r[k];  // 新节点右侧 是原k的右侧
    l[r[k]] = tot;  // 原k右侧的左侧 是新节点
    r[k] = tot;     // k右侧 是新节点
}

void del(int k) {
    r[l[k]] = r[k];
    l[r[k]] = l[k];
}

int main() {
    string c;
    int m, k, x;
    init();
    cin >> m;
    while(m--) {
        cin >> c;
        if(c == "L") {
            cin >> x;
            addl(x);
        } else if(c == "R") {
            cin >> x;
            addr(x);
        } else if(c == "D") {
            cin >> k;
            del(k+2);       // 从3开始存 故+2
        } else if(c == "IL") {
            cin >> k >> x;
            addkl(k+2, x);
        } else if(c == "IR") {
            cin >> k >> x;
            addkr(k+2, x);
        }
    }
    for(int i = r[0]; i != 1; i = r[i]) cout <<node[i]<<" ";
    cout <<endl;
}
```

##### **Python**

```python


 #删除第k个插入的数
def remove(k):
    r[l[k]] = r[k]
    l[r[k]] = l[k]

 #在节点k的右边插入一个数x
def insert(k, x):
    global idx
    ev[idx] = x
    l[idx], r[idx] = k, r[k]
    l[r[k]] = idx
    r[k] = idx
    idx += 1
    

if __name__ == '__main__':

    N = 100000 
    n = int(input())

    #下标是0的点是head，下标是1的点是tail
    ev=[0]*N;l=[0]*N;r=[0]*N
    r[0]=1;l[1]=0;idx=2
    
    for i in range(n):
        op = input().split()
        if op[0] == 'L':
            x = int(op[1])
            insert(0, x)

        elif op[0] == 'R':
            x = int(op[1])
            insert(l[1], x)

        elif op[0] == 'D':
            k = int(op[1])
            remove(k+1)

        elif op[0] == 'IL':
            k, x = int(op[1]), int(op[2])
            insert(l[k+1], x)

        else:
            k, x = int(op[1]), int(op[2])
            insert(k+1, x)

    p = r[0]
    res = []
    while p != 1:
        res.append(ev[p])
        p = r[p]

    print(' '.join(map(str, res)))
```

<!-- tabs:end -->
</details>

<br>

* * *

### 链表操作

> [!NOTE] **[LeetCode 19. 删除链表的倒数第N个节点](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode *dummy = new ListNode(-1);
        dummy->next = head;
        ListNode *fast = dummy, *slow = dummy;
        while (n -- ) fast = fast->next;
        while (fast->next) {
            fast = fast->next;
            slow = slow->next;
        }
        slow->next = slow->next->next;
        return dummy->next;
    }
};
```

##### **Python**

```python
# 删除的点可能是head节点，所以一定要用dummy方便处理头节点被删的情况

class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dummy = ListNode(None)
        dummy.next = head 
        p1, p2 = dummy, dummy
        for _ in range(n):
            p2 = p2.next 
        while p2 and p2.next:
            p1 = p1.next 
            p2 = p2.next
        p1.next = p1.next.next   # 如果这里写p1.next = p2 会有[1] 1这个case过不了
        # 在这个case下， p1还是在dummy处没动，p2也在第一个点，所以不能直接p1.next = p2
        return dummy.next
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 21. 合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode* dummy = new ListNode(-1);
        ListNode* pre = dummy;
        while (l1 && l2) {
            if (l1->val <= l2->val) {
                pre->next = l1;
                l1 = l1->next;
            } else {
                pre->next = l2;
                l2 = l2->next;
            }
            pre = pre->next;
        }
        l1 ? pre->next = l1 : pre->next = l2;
        return dummy->next;
    }
};
```

##### **Python**

```python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = ListNode(None)
        p = dummy
        while l1 and l2:
            if l1.val < l2.val:
                p.next = l1
                l1 = l1.next
            else:
                p.next = l2
                l2 = l2.next
            p = p.next
        p.next = l1 if l1 else l2
        return dummy.next
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 23. 合并K个升序链表](https://leetcode.cn/problems/merge-k-sorted-lists/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:

    struct Cmp {    // ATTENTION
        bool operator() (ListNode* a, ListNode* b) {
            return a->val > b->val;
        }
    };

    ListNode* mergeKLists(vector<ListNode*>& lists) {
        priority_queue<ListNode*, vector<ListNode*>, Cmp> heap;
        auto dummy = new ListNode(-1), tail = dummy;
        for (auto l : lists) if (l) heap.push(l);

        while (heap.size()) {
            auto t = heap.top();
            heap.pop();

            tail = tail->next = t;
            if (t->next) heap.push(t->next);
        }

        return dummy->next;
    }
};
```

##### **C++ 2**

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    struct cmp {
        bool operator()(ListNode* a, ListNode* b) {
            return a->val > b->val;
        }
    };
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        int k = lists.size();
        priority_queue<ListNode*, vector<ListNode*>, cmp> pq;
        for (auto p : lists) if (p) pq.push(p);   // if(p)很重要
        ListNode* dummy = new ListNode(-1);
        ListNode* pre = dummy;
        while (!pq.empty()) {
            ListNode* t = pq.top(); pq.pop();
            pre->next = t;
            pre = t;
            if (t->next) pq.push(t->next);
        }
        return dummy->next;
    }
};
```

##### **Python**

```python
# 1. 用小根堆存储k个排序链表的头指针，并且把每个链表的头指针往后移一位
# 2. 当小根堆不为空的时候，弹出最小的元素，p的next指针指向那个节点，并且把最小的那条链表的下一个头节点加入到堆中

import heaqp
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        q = []
        n = len(lists)
        for i in range(n):
            # 注意输入为 [[]] 的case
            if lists[i]:
                heapq.heappush(q, (lists[i].val, i))
                lists[i] = lists[i].next

        dummy = ListNode(None)
        pre = dummy
        while q:
            val, idx = heapq.heappop(q)
            pre.next = ListNode(val)
            pre = pre.next 
            if lists[idx]:
                heapq.heappush(q, (lists[idx].val, idx))
                lists[idx] = lists[idx].next
        return dummy.next
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 24. 两两交换链表中的节点](https://leetcode.cn/problems/swap-nodes-in-pairs/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 迭代**

```cpp
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        auto dummy = new ListNode(-1);
        dummy->next = head;
        for (auto p = dummy; p->next && p->next->next;) {
            auto a = p->next, b = a->next;
            p->next = b;
            a->next = b->next;
            b->next = a;
            p = a;
        }

        return dummy->next;
    }
};

class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        ListNode *dummy = new ListNode(-1);
        dummy->next = head;
        ListNode *pre = dummy, *cur = head, *next;
        while (cur && cur->next) {
            next = cur->next;
            cur->next = next->next;
            next->next = cur;
            pre->next = next;

            pre = cur;
            cur = cur->next;
        }
        return dummy->next;
    }
};
```

##### **C++ 递归**

```cpp
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        if (!head || !head->next) return head;
        ListNode *second = head->next;
        head->next = swapPairs(second->next);
        second->next = head;
        return second;
    }
};
```

##### **Python**

```python
# 添加虚拟头结点 dummy。定义 pre 指针初始指向 dummy。
# 定义 cur 指向 head, 若 cur 或 cur.next 为空，则终止循环。
# 按照一定的次序，修改 pNe 和 cur 的 next 指针，具体参见代码。

class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        dummy = ListNode(None)
        pre = dummy
        # 踩坑！初始化pre.next = head， 可以覆盖到一些特殊case:[], [1]
        pre.next = head
        cur = head 
        # 踩坑！ 这里要加上 判断cur.next 
        while cur and cur.next:  
            # 不然一进来，可能pNe就是None，
            pNe = cur.next    
            # 就会导致pNe没有next, 而报错
            cur.next = pNe.next   
            pNe.next = cur 
            pre.next = pNe
            pre, cur = cur, cur.next
        return dummy.next
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 25. K 个一组翻转链表](https://leetcode.cn/problems/reverse-nodes-in-k-group/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        ListNode* dummy = new ListNode(-1);
        dummy->next = head;
        ListNode *pre = dummy, *cur = head, *next;
        while (cur) {
            ListNode *tail = pre;
            for (int i = 0; i < k; ++i) {
                tail = tail->next;
                if (!tail) return dummy->next;
            }
            // tail 是这k个的最后一个
            next = tail->next;
            // 翻转
            ListNode *fpre = tail->next, *fcur = pre->next, *fnext;
            while (fcur != next) {
                fnext = fcur->next;
                fcur->next = fpre;
                fpre = fcur;
                fcur = fnext;
            }
            pre->next = fpre;
            pre = cur;
            cur = next;
        }
	    return dummy->next;
    }
};

class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        auto dummy = new ListNode(-1);
        dummy->next = head;
        for (auto p = dummy;;) {
            auto q = p;
            for (int i = 0; i < k && q; i ++ ) q = q->next;
            if (!q) break;
            auto a = p->next, b = a->next;
            for (int i = 0; i < k - 1; i ++ ) {
                auto c = b->next;
                b->next = a;
                a = b, b = c;
            }
            auto c = p->next;
            p->next = a, c->next = b;
            p = c;
        }
        return dummy->next;
    }
};
```

##### **Python**

```python
# 1. 先暴力判断是否有k个节点；2. 内部交换，最后再换头尾
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        dummy = ListNode(-1)
        dummy.next = head 
        pre = dummy
        while pre:
            q = pre 
            i = 0
            while i < k and q: # 注意 不能修改k, 所有用i变量来暴力遍历！ 踩坑！这里需要把 q是否存在加入到判断条件里，不会q.next就会报错！
                q = q.next 
                i += 1
            if not q:break 

            cur = pre.next 
            pNe = cur.next 
            for _ in range(k - 1):
                tmp = pNe.next 
                pNe.next = cur 
                cur, pNe = pNe, tmp
            
            a = pre.next   # 踩坑：这一部分 也记得要更新pre的位置，所以需要借助一个a
            pre.next = cur 
            a.next = pNe 
            pre = a 
        return dummy.next
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 61. 旋转链表](https://leetcode.cn/problems/rotate-list/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
class Solution {
public:
    ListNode* rotateRight(ListNode* head, int k) {
        if (!head) return head;
        int n = 0;
        ListNode* tail;
        for (auto p = head; p; p = p->next) {
            tail = p;
            n ++ ;
        }
        k %= n;
        if (!k) return head;

        auto p = head;
        for (int i = 0; i < n - k - 1; i ++ ) p = p->next;
        tail->next = head;
        head = p->next;
        p->next = nullptr;
        return head;
    }
};
```

##### **C++ 2**

```cpp
class Solution {
public:
    ListNode* rotateRight(ListNode* head, int k) {
        ListNode* t = head;
        int tot = 0;
        while (t) t = t->next, ++ tot ;
        // if (!tot || tot == 1 || tot%k == 0) return head;
        if (tot <= 1 || k%tot == 0) return head;
        k %= tot;
        ListNode *slow = head, *fast = head;
        while (k -- ) fast = fast->next;
        while (fast->next) {
            slow = slow->next;
            fast = fast->next;
        }
        fast->next = head;
        t = slow->next;
        slow->next = nullptr;
        return t;
    }
};
```

##### **Python**

```python
"""
这道题中 k 可能很大，所以我们令 k=k%n，n是链表长度。
1. 创建两个指针first, second，分别指向头结点
2. 先让first向后移动 k 个位置，然后first和second同时向后移动，直到first走到链表最后一个元素。
3. 此时first指向链表末尾，second指向分界点。然后我们把链表从分界点处断开，然后把后半段接在前半段前面即可。

"""
class Solution:
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        if not head:return 
        dummy = ListNode(None)
        pre = dummy
        dummy.next = head 
        length = 0
        while pre.next:
            pre = pre.next 
            length += 1
        p1, p2, k = head, head, k % length
        if k == 0:return head
        for _ in range(k):
            p2 = p2.next 
        while p2.next:
            p1 = p1.next
            p2 = p2.next 
        dummy.next = p1.next
        p1.next = None
        p2.next = head
        return dummy.next
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 82. 删除排序链表中的重复元素 II](https://leetcode.cn/problems/remove-duplicates-from-sorted-list-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    // 注意 1 1这种特殊情况
    ListNode* deleteDuplicates(ListNode* head) {
        ListNode *dummy = new ListNode(-1);
        dummy->next = head;
        ListNode *slow = dummy, *fast = dummy->next;
        while (fast != nullptr) {
            if (fast->next != nullptr && (fast->val == fast->next->val)) {
                int tmpv = fast->val;
                while (fast != nullptr && fast->val == tmpv)
                    fast = fast->next;
            } else {
                slow->next = fast;    // attention
                slow = fast;
                fast = fast->next;
            }
        }
        slow->next = fast;
        return dummy->next;
    }
};


// yxc
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        ListNode* dummy = new ListNode(-1);
        dummy->next = head;
        ListNode* p = dummy;
        while (p->next) {
            ListNode* q = p->next;
            while (q && q->val == p->next->val)
                q = q->next;
            if (p->next->next == q) p = p->next;
            else p->next = q;
        }
        return dummy->next;
    }
};
```

##### **Python**

```python
# 法一：用一个哨兵 来判断pre当前数 能不能被用
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        dummy = ListNode(None)
        pre = dummy
        dummy.next = head 
        cur = head 
        while cur:
            flag = False
            while cur.next:
                if cur.val == cur.next.val:
                    cur = cur.next
                    flag = True
                else:break
            if flag:
                pre.next = cur.next 
            else:
                pre = cur 
            cur = cur.next 
        return dummy.next
      
# 法二：
"""
为了方便处理边界情况，我们定义一个虚拟元素 dummy 指向链表头节点。
然后从前往后扫描整个链表，每次扫描元素相同的一段，如果这段中的元素个数多于1个，则将整段元素直接删除。

时间复杂度分析：整个链表只扫描一遍，所以时间复杂度是 O(n。
"""
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 86. 分隔链表](https://leetcode.cn/problems/partition-list/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
class Solution {
public:
    ListNode* partition(ListNode* head, int x) {
        auto lh = new ListNode(-1), rh = new ListNode(-1);
        auto lt = lh, rt = rh;

        for (auto p = head; p; p = p->next) {
            if (p->val < x) lt = lt->next = p;
            else rt = rt->next = p;
        }

        lt->next = rh->next;
        rt->next = NULL;

        return lh->next;
    }
};
```

##### **C++ 2**

```cpp
class Solution {
public:
    ListNode* partition(ListNode* head, int x) {
        ListNode *before = new ListNode(0);
        ListNode *befHead = before;
        ListNode *after = new ListNode(0);
        ListNode *aftHead = after;
        while (head != nullptr) {
            if (head->val < x) {
                before->next = head;
                before = before->next;
            } else {
                after->next = head;
                after = after->next;
            }
            head = head->next;
        }
        after->next = nullptr;
        before->next = aftHead->next;
        return befHead->next;
    }
};
```

##### **Python**

```python
# 两条链表：一条链表存储 比 x 小的节点；另外一条存比 x 大的节点；再把两个链表拼接起来

class Solution:
    def partition(self, head: ListNode, x: int) -> ListNode:
        dum1, dum2 = ListNode(None), ListNode(None)
        p1, p2 = dum1, dum2
        p = head 
        while p:
            if p.val < x:
                p1.next = p 
                p1 = p1.next 
            else:
                p2.next = p 
                p2 = p2.next 
            p = p.next 
        p1.next = dum2.next 
        p2.next = None
        return dum1.next
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 92. 反转链表 II](https://leetcode.cn/problems/reverse-linked-list-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* reverseBetween(ListNode* head, int left, int right) {
        int d = right - left + 1;
        if (d <= 1)
            return head;
        
        ListNode * dummy = new ListNode(-1);
        dummy->next = head;
        
        ListNode * before = dummy;
        while ( -- left)    // ATTENTION
            before = before->next;

        // 设置pre 注意需要d+1个位移
        ListNode * pre = before;
        for (int i = 0; i <= d; ++ i )
            pre = pre->next;
        ListNode * cur = before->next;
        while (d -- ) {
            auto next = cur->next;
            cur->next = pre;

            pre = cur;
            cur = next;
        }
        before->next = pre;
        return dummy->next;
    }
};
```

##### **C++ 2**

```cpp
// yxc
class Solution {
public:
    ListNode* reverseBetween(ListNode* head, int m, int n) {
        auto dummy = new ListNode(-1);
        dummy->next = head;

        auto a = dummy;
        for (int i = 0; i < m - 1; i ++ ) a = a->next;
        auto b = a->next, c = b->next;
        for (int i = 0; i < n - m; i ++ ) {
            auto t = c->next;
            c->next = b;
            b = c, c = t;
        }
        a->next->next = c;
        a->next = b;
        return dummy->next;
    }
};
```
##### **Python**

```python
class Solution:
    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
        dum = ListNode(None)
        dum.next = head
        p1, p2 = dum, dum
        while m > 0:
            pre1 = p1
            p1 = p1.next
            m -= 1
        while n > 0:
            pre2 = p2
            p2 = p2.next
            n -= 1
        p2N = p2.next
        pre, cur = p2N, p1
        while cur != p2N:
            tmp = cur.next
            cur.next = pre
            pre = cur
            cur = tmp
        pre1.next = pre
        return dum.next
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 138. 复制带随机指针的链表](https://leetcode.cn/problems/copy-list-with-random-pointer/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    Node* copyRandomList(Node* head) {
        for (auto p = head; p; p = p->next->next) {  // 复制一个小弟
            auto q = new Node(p->val);
            q->next = p->next;
            p->next = q;
        }

        // 复制random指针
        for (auto p = head; p; p = p->next->next)
            if (p->random)
                p->next->random = p->random->next;

        // 拆分两个链表
        auto dummy = new Node(-1), pre = dummy;
        for (auto p = head; p; p = p->next) {
            auto q = p->next;
            pre = pre->next = q;
            p->next = q->next;
        }
        return dummy->next;
    }
};
```

##### **Python**

```python
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head:return head

        p = head    # 第一步：复制一个小弟
        while p:
            q = Node(p.val)
            q.next = p.next 
            p.next = q 
            p = p.next.next 

        p = head  # 第二步：赋值random指针
        while p:
            if p.random:
                p.next.random = p.random.next 
            p = p.next.next

        dummy = ListNode(None)  # 第三步：拆分
        cur = dummy
        p = head 
        while p:
            cur.next = p.next
            cur = cur.next 
            p = p.next.next
        return dummy.next
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 141. 环形链表](https://leetcode.cn/problems/linked-list-cycle/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    bool hasCycle(ListNode *head) {
        ListNode *fast = head, *slow = head;
        while (fast && fast->next) {
            fast = fast->next->next;
            slow = slow->next;
            if (slow == fast) return true;
        }
        return false;
    }
};
```

##### **Python**

```python
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        if not head or not head.next:return False
        slow, fast = head, head 
        while fast and fast.next: # 注意 这里是fast.next! 而不是fast.next.next 当只有一个节点时。
            pF = pF.next.next
            slow = slow.next 
            fast = fast.next.next 
            if slow == fast:return True 
        return False      
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 142. 环形链表 II](https://leetcode.cn/problems/linked-list-cycle-ii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        ListNode *fast = head, *slow = head, *ret = nullptr;
        while (fast && fast->next) {
            fast = fast->next->next;
            slow = slow->next;
            if (fast == slow) {
                // fast 比 slow 多走了 x 个环的长度
                ret = head;
                while (ret != slow) {
                    ret = ret->next;
                    slow = slow->next;
                }
                break;
            }
        }
        return ret;
    }
};
```

##### **Python**

```python
class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        if not head or not head.next:return None 
        fast, slow = head, head 
        while fast and fast.next:
            slow = slow.next 
            fast = fast.next.next 
            if slow == fast:
                fast = head 
                while slow != fast:
                    fast = fast.next 
                    slow = slow.next 
                return slow
        return None   # 记得最后如果没有环 返回None
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 143. 重排链表](https://leetcode.cn/problems/reorder-list/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 1**

```cpp
// yxc
class Solution {
public:
    void reorderList(ListNode* head) {
        if (!head) return;
        int n = 0;
        for (auto p = head; p; p = p->next) n ++ ;

        auto mid = head;
        for (int i = 0; i < (n + 1) / 2 - 1; i ++ ) mid = mid->next;
        auto a = mid, b = a->next;
        for (int i = 0; i < n / 2; i ++ ) {
            auto c = b->next;
            b->next = a, a = b, b = c;
        }
        auto p = head, q = a;
        for (int i = 0; i < n / 2; i ++ ) {
            auto o = q->next;
            q->next = p->next;
            p->next = q;
            p = q->next, q = o;
        }

        if (n % 2) mid->next = NULL;
        else mid->next->next = NULL;
    }
};
```

##### **C++ 2**

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    void reorderList(ListNode* head) {
        if (!head) return;
        ListNode *slow = head, *fast = head;
        // 这种写法 结束时 slow在中点上或中点隔板左侧
        while (fast->next && fast->next->next) {
            slow = slow->next;
            fast = fast->next->next;
        }
        // head后的长度与nhead长度相等 或比其大1
        ListNode *nhead = slow->next;
        slow->next = nullptr;

        ListNode *pre = nullptr, *cur = nhead, *next;
        while (cur) {
            next = cur->next;
            cur->next = pre;
            pre = cur;
            cur = next;
        }

        ListNode *next2;
        cur = pre, pre = head;
        while (cur) {
            next = pre->next;
            next2 = cur->next;
            pre->next = cur;
            cur->next = next;
            pre = next;
            cur = next2;
        }
        return;
    }
};
```

##### **Python**

```python
# 1. 找中点前一个节点 2. 翻转后半部分 3. 拆分插入
class Solution:
    def reorderList(self, head: ListNode) -> None:
        if not head or not head.next:return head 
        fast, slow = head, head
        while fast.next and fast.next.next:
            fast = fast.next.next 
            slow = slow.next 
        pre = None 
        cur = slow.next 
        p = head 
        while cur:
            tmp = cur.next
            cur.next = pre 
            pre = cur 
            cur = tmp 
        slow.next = pre
        p1, p2 = head, slow.next
        while p1 != slow:
            p1_tmp, p2_tmp = p1.next, p2.next 
            p1.next = p2
            p2.next = p1_tmp 
            p1, p2 = p1_tmp, p2_tmp 
        if p2:
            p1.next = p2 
            p2.next = None 
        else:
            p1.next = None
        return head
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 147. 对链表进行插入排序](https://leetcode.cn/problems/insertion-sort-list/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    ListNode* insertionSortList(ListNode* head) {
        ListNode *dummy = new ListNode(INT_MIN);
        ListNode *pre = dummy;
        while (head) {
            // 令 pre = dummy 规避对pre是否为空的判断
            // 这里多了个 pre 是为了一定程度上加速
            ListNode * p = nullptr;
            if (pre->val <= head->val) p = pre;
            else p = dummy;
            while (p->next && p->next->val < head->val)
                p = p->next;

            auto next = head->next;
            head->next = p->next;
            p->next = head;
            pre = head;
            
            head = next;
        }
        return dummy->next;
    }
};

class Solution {
public:
    ListNode* insertionSortList(ListNode* head) {
        ListNode *dummy = new ListNode(-1);
        while (head) {
            ListNode *p = dummy;
            while (p->next && p->next->val <= head->val) p = p->next;

            ListNode *next = head->next;
            head->next = p->next;
            p->next = head;

            head = next;
        }
        return dummy->next;
    }
};
```

##### **Python**

```python
# 
class Solution:
    def insertionSortList(self, head: ListNode) -> ListNode:
        dummy = ListNode(None)
        p = head
        while p:   # p指向要插入的节点
            cur = dummy 
            pNe = p.next
            while cur.next and cur.next.val <= p.val:
                cur = cur.next
            p.next = cur.next 
            cur.next = p 
            p = pNe
        return dummy.next
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 148. 排序链表](https://leetcode.cn/problems/sort-list/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 迭代优雅写法
> 
> 滚瓜烂熟: 递归 + 迭代

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 标准 递归**

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode * merge(ListNode * l, ListNode * r) {
        ListNode * dummy = new ListNode(INT_MIN);
        auto cur = dummy;   // ATTENTION
        while (l && r)
            if (l->val <= r->val)
                cur = cur->next = l, l = l->next;
            else
                cur = cur->next = r, r = r->next;
        // while (l)
        //     cur = cur->next = l, l = l->next;
        // while (r)
        //     cur = cur->next = r, r = r->next;
        // 也可以省略重复操作
        cur = cur->next = (l ? l : r);
        
        return dummy->next;
    }

    ListNode* sortList(ListNode* head) {
        if (head == nullptr || head->next == nullptr)
            return head;    // ATTENTION head
        
        // 这种写法 当循环结束时
        // - 奇数长度: slow为中点
        // - 偶数长度: slow为前一段的末尾
        ListNode * slow = head, * fast = head;
        while (fast->next && fast->next->next)
            slow = slow->next, fast = fast->next->next;
        
        // p, q 分别为两段的起始
        ListNode * p = head, * q = slow->next;
        slow->next = nullptr;   // 截断

        ListNode * left = sortList(p), * right = sortList(q);
        return merge(left, right);
    }
};
```

##### **C++ 标准 迭代**

```cpp
class Solution {
public:
    ListNode* sortList(ListNode* head) {
        int n = 0;
        for (auto p = head; p; p = p->next) n ++ ;
        
        ListNode * dummy = new ListNode(INT_MIN);
        dummy->next = head;

        for (int d = 1; d < n; d <<= 1 ) {
            auto cur = dummy;
            for (int i = 1; i + d <= n; i += d * 2) {   // 细节: i本身只用作计数 流程中不会被使用
                // 前半段的起点 后半段的起点
                auto p = cur->next, q = p;
                for (int k = 0; k < d; ++ k ) q = q->next;

                int x = 0, y = 0;
                while (x < d && y < d && p && q)
                    if (p->val <= q->val)
                        cur = cur->next = p, p = p->next, x ++ ;
                    else
                        cur = cur->next = q, q = q->next, y ++ ;
                
                while (x < d && p)
                    cur = cur->next = p, p = p->next, x ++ ;
                while (y < d && q)
                    cur = cur->next = q, q = q->next, y ++ ;
                
                // 结束后 q处于新的一段的起始
                cur->next = q;
            }
        }
        return dummy->next;
    }
};
```

##### **C++ 迭代 yxc**

```cpp
class Solution {
public:
    ListNode* sortList(ListNode* head) {
        int n = 0;
        for (auto p = head; p; p = p->next) n ++ ;

        auto dummy = new ListNode(-1);
        dummy->next = head;
        for (int i = 1; i < n; i *= 2) {
            auto cur = dummy;
            for (int j = 1; j + i <= n; j += i * 2) {
                auto p = cur->next, q = p;
                for (int k = 0; k < i; k ++ ) q = q->next;
                int x = 0, y = 0;
                while (x < i && y < i && p && q) {
                    if (p->val <= q->val) cur = cur->next = p, p = p->next, x ++ ;
                    else cur = cur->next = q, q = q->next, y ++ ;
                }
                while (x < i && p) cur = cur->next = p, p = p->next, x ++ ;
                while (y < i && q) cur = cur->next = q, q = q->next, y ++ ;
                cur->next = q;
            }
        }

        return dummy->next;
    }
};
```

##### **C++ 旧 分治归并**

```cpp
class Solution {
public:
    ListNode* sortList(ListNode* head) {
        ListNode *dummy = new ListNode(INT_MIN);
        dummy->next = head;
        ListNode *p = head;
        int len = 0;
        while (p) {
            ++ len ;
            p = p->next;
        }
        for (int d = 1; d < len; d <<= 1) {
            ListNode *pre = dummy, *cur = dummy->next;
            while (cur) {
                ListNode *l = cur;
                ListNode *r = cut(l, d);
                cur = cut(r, d);
                pre->next = merge(l, r);
                while (pre->next) pre = pre->next;
            }
        }
        return dummy->next;
    }
    // 切掉前d个返回新头节点
    ListNode* cut(ListNode* n, int d) {
        ListNode *p = n, *next;
        while ( -- d && p) {
            p = p->next;
        }
        if (!p) return nullptr;
        next = p->next;
        p->next = nullptr;
        return next;
    }
    ListNode* merge(ListNode* l1, ListNode* l2) {
        ListNode* dummy = new ListNode(-1);
        ListNode* p = dummy;
        while (l1 && l2) {
            if (l1->val < l2->val) {
                p->next = l1;
                l1 = l1->next;
            } else {
                p->next = l2;
                l2 = l2->next;
            }
            p = p->next;
        }
        p->next = l1 ? l1 : l2;
        return dummy->next;
    }
};
```

##### **Python**

```python
# 只需要掌握 递归版的归并排序

#  迭代的归并排序空间复杂度是S(1); 递归的归并排序:S(logN)
# 快排的空间复杂度S(logN)

# 选择 迭代的归并排序：自底向上做！（正常递归的归并排序是 自顶向下）
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        if not head or not head.next: 
            return head
        slow, fast = head, head
        while fast.next and fast.next.next:  # 踩坑：这里不能写fast and fast.next， 当链表只有两个节点的时候，就会陷入死循环，一直递归
            slow = slow.next
            fast = fast.next.next
        # 找到左右部分, 把左部分最后置空
        mid = slow.next
        slow.next = None
        # 递归下去
        left = self.sortList(head)
        right = self.sortList(mid)
        # 合并
        return self.merge(left, right)

    def merge(self, left, right):
        dummy = ListNode(0)
        p = dummy
        l, r = left, right

        while l and r:
            if l.val < r.val:
                p.next = l
                l = l.next
                p = p.next
            else:
                p.next = r
                r = r.next
                p = p.next
        if l:
            p.next = l
        if r:
            p.next = r
        return dummy.next
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 160. 相交链表](https://leetcode.cn/problems/intersection-of-two-linked-lists/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 双指针思想

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        ListNode *l1 = headA, *l2 = headB;
        while (l1 != l2) {
            l1 = l1 ? l1->next : headB;
            l2 = l2 ? l2->next : headA;
        }
        return l1;
    }
};
```

##### **Python**

```python
# 不管是相交还是 不相交，跳出循环的时候 p1 == p2
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        if not headA or not headB:return None
        p1, p2 = headA, headB
        while p1 != p2:
            p1 = p1.next if p1 else headB 
            p2 = p2.next if p2 else headA
        return p1  
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 203. 移除链表元素](https://leetcode.cn/problems/remove-linked-list-elements/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ yxc**

```cpp
// yxc
class Solution {
public:
    ListNode* removeElements(ListNode* head, int val) {
        auto dummy = new ListNode(-1);
        dummy->next = head;
        for (auto p = dummy; p; p = p->next) {
            auto q = p->next;
            while (q && q->val == val) q = q->next;
            p->next = q;
        }
        return dummy->next;
    }
};
```

##### **C++**

```cpp
class Solution {
public:
    ListNode* removeElements(ListNode* head, int val) {
        ListNode *dummy = new ListNode(-1);
        dummy->next = head;
        ListNode *pre = dummy;
        while (pre) {
            // 1.
            // if (pre->next && pre->next->val == val) pre->next = pre->next->next;
            // else pre = pre->next;
            // 2.
            auto p = pre->next;
            while (p && p->val == val) p = p->next;
            pre->next = p;
            pre = p;
        }
        return dummy->next;
    }
};
```

##### **Python**

```python
class Solution:
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        dummy=ListNode(None)
        dummy.next=head
        pre=dummy
        cur=pre.next
        while cur:
            if cur.val==val:
                pre.next=cur.next
                cur=pre.next
            else:
                cur=cur.next
                pre=pre.next
        return dummy.next
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 206. 反转链表](https://leetcode.cn/problems/reverse-linked-list/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode *prev = nullptr;
        while (head) {
            auto next = head->next;
            head->next = prev;
            prev = head;
            head = next;
        }
        return prev;
    }

    ListNode* reverseList(ListNode* head) {
        ListNode* dummy = new ListNode(-1);
        dummy->next = head;
        ListNode* pre = nullptr, * cur = head;
        while (cur != nullptr) {
            auto next = cur->next;
            cur->next = pre;
            pre = cur;
            cur = next;
        }
        dummy->next = pre;
        return dummy->next;
    }
};

// 递归
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if (!head || !head->next) return head;
        ListNode *tail = reverseList(head->next);
        head->next->next = head;
        head->next = nullptr;
        return tail;
    }
};
```

##### **Python**

```python
# python3
#========= 迭代
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# 迭代
# 1. 用一个 cur 指针指向 head头节点，pre指针指向None， tmp指针用于保存 cur.next节点
# 2. 循环遍历 cur， 当 cur 指针存在时，首先先把 cur.next节点暂存在 tmp 指针中，然后把 cur.next 指向 pre。 pre 指针 移到 cur 指针的位置；最后 把 cur 指向 之前暂存的 tmp指针。
# 3. 最后当 cur 空时，跳出循环，此时 pre 指向的是 原链表的尾节点。所以返回 pre即可。

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if not head or not head.next:return 
        cur, pre, p = head, None, None
        while cur:
            p = cur.next
            cur.next = pre
            pre = cur 
            cur = p
        return pre


#========= 递归
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
      	 # 踩坑！
        if not head or not head.next: 
            return head
        # 这里最终递归返回的是尾节点
        cur = self.reverseList(head.next) 
        # 对于任意一个节点，将它的next指向他本身
        head.next.next = head   
        # 最后将head.next置为None 
        head.next = None  
         # 所以将它输出即可
        return cur 
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 234. 回文链表](https://leetcode.cn/problems/palindrome-linked-list/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    bool isPalindrome(ListNode* head) {
        if (!head || !head->next) return true;
        ListNode *slow = head, *fast = head, *pre = nullptr, *cur = head;
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;

            auto next = cur->next;
            cur->next = pre;
            pre = cur;
            cur = next;
        }
        if (fast) slow = slow->next;
        while (pre && slow) {
            if (pre->val != slow->val)
                return false;
            pre = pre->next;
            slow = slow->next;
        }
        return true;
    }
};
```

##### **C++ yxc**

```cpp
// yxc
class Solution {
public:
    bool isPalindrome(ListNode* head) {
        int n = 0;
        for (auto p = head; p; p = p->next) n ++ ;
        if (n <= 1) return true;
        int half = n / 2;
        auto a = head;
        for (int i = 0; i < n - half; i ++ ) a = a->next;
        auto b = a->next;
        for (int i = 0; i < half - 1; i ++ ) {
            auto c = b->next;
            b->next = a;
            a = b, b = c;
        }

        auto p = head, q = a;
        bool success = true;
        for (int i = 0; i < half; i ++ ) {
            if (p->val != q->val) {
                success = false;
                break;
            }
            p = p->next;
            q = q->next;
        }

        auto tail = a;
        b = a->next;
        // 将链表恢复原状
        for (int i = 0; i < half - 1; i ++ ) {
            auto c = b->next;
            b->next = a;
            a = b, b = c;
        }

        tail->next = NULL;
        return success;
    }
};
```

##### **Python**

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        if not head or not head.next:return True
        dummy=ListNode(None)
        dummy.next=head
        p1,p2=dummy,dummy
        stack=[]
        while p2 and p2.next:
            p1,p2=p1.next,p2.next.next
        p1=p1.next
        while p1:
            stack.append(p1)
            p1=p1.next
        p=head
        while stack:
            q=stack.pop()
            if p.val!=q.val:
                return False
            p=p.next
        return True
            

        
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 237. 删除链表中的节点](https://leetcode.cn/problems/delete-node-in-a-linked-list/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    void deleteNode(ListNode* node) {
        *node = *node->next;
    }
    void deleteNode_2(ListNode* node) {
        node->val = node->next->val;
        node->next = node->next->next;
    }
};
```

##### **Python**

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val, node.next = node.next.val, node.next.next
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 328. 奇偶链表](https://leetcode.cn/problems/odd-even-linked-list/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    ListNode* oddEvenList(ListNode* head) {
        ListNode *dummy1 = new ListNode(-1), *dummy2 = new ListNode(-1);
        ListNode *p1 = dummy1, *p2 = dummy2;
        bool giveA = true;
        while (head) {
            if (giveA) p1->next = head, p1 = head;
            else p2->next = head, p2 = head;
            head = head->next;
            giveA = !giveA;
        }
        p2->next = nullptr;
        p1->next = dummy2->next;
        return dummy1->next;
    }
};
```

##### **C++ 标准**

```cpp
class Solution {
public:
    ListNode* oddEvenList(ListNode* head) {
        if (!head || !head->next) return head;
        auto oh = head, ot = oh;
        auto eh = head->next, et = eh;
        for (auto p = head->next->next; p;) {
            ot = ot->next = p;
            p = p->next;
            if (p) {
                et = et->next = p;
                p = p->next;
            }
        }
        ot->next = eh;
        et->next = NULL;
        return oh;
    }
};
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 725. 分隔链表](https://leetcode.cn/problems/split-linked-list-in-parts/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    vector<ListNode*> splitListToParts(ListNode* root, int k) {
        int n = 0;
        for (auto p = root; p; p = p->next) n ++ ;

        vector<ListNode*> res;
        auto p = root;
        for (int i = 0; i < k; i ++ ) {
            int len = n / k;
            if (i < n % k) len ++ ;
            res.push_back(p);
            for (int j = 0; j < len - 1; j ++ ) p = p->next;
            if (p) {
                auto q = p->next;
                p->next = NULL;
                p = q;
            }
        }
        return res;
    }
};
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1171. 从链表中删去总和值为零的连续节点](https://leetcode.cn/problems/remove-zero-sum-consecutive-nodes-from-linked-list/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 扫一遍即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    ListNode* removeZeroSumSublists(ListNode* head) {
        unordered_map<int, ListNode*> hash; // mem the pre
        ListNode * dummy = new ListNode(-1);
        dummy->next = head;
        int sum = 0;
        hash[0] = dummy;
        
        auto p = head, last = dummy;
        while (p) {
            sum += p->val;
            last = p;
            if (hash.count(sum)) {
                int tsum = 0;
                auto pre = hash[sum];
                auto first = pre->next;
                while (first != p->next) {
                    tsum += first->val;
                    hash.erase(sum + tsum);
                    first = first->next;
                }
                last = pre;
                pre->next = p->next;
            }
            hash[sum] = last;
            p = p->next;
        }
        return dummy->next;
    }
};
```

##### **C++ 更简约**

以前的写法，更简约清晰：

```cpp
class Solution {
public:
    ListNode* removeZeroSumSublists(ListNode* head) {
        unordered_map<int, ListNode*> hash;     // 记录当前值节点
        ListNode * dummy = new ListNode(-1);
        dummy->next = head;
        int sum = 0;
        hash[0] = dummy;

        while(head) {
            sum += head->val;
            if(hash.count(sum)) {
                auto next = hash[sum]->next;
                hash[sum]->next = head->next;   // 在这里处理 更简约
                int csum = sum;
                while(next != head) {
                    csum += next->val;
                    hash.erase(csum);
                    next = next->next;
                }
                // 不管next->next
            } else hash[sum] = head;            // 而非last
            head = head->next;
        }
        return dummy->next;
    }
};
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 2095. 删除链表的中间节点](https://leetcode.cn/problems/delete-the-middle-node-of-a-linked-list/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 坑
> 
> - 本题是中间靠右且要找中间的前一个 所以 `while (fast->next && fast->next->next)`
> 
> - dummy->next

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* deleteMiddle(ListNode* head) {
        ListNode * dummy = new ListNode(-1);
        dummy->next = head;
        auto slow = dummy, fast = dummy;
        while (fast->next && fast->next->next)
            fast = fast->next->next, slow = slow->next;
        if (slow->next)
            slow->next = slow->next->next;
        return dummy->next;
    }
};
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 708. 循环有序列表的插入](https://leetcode.cn/problems/insert-into-a-sorted-circular-linked-list/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 细节 找前后区间 while判断条件

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* next;

    Node() {}

    Node(int _val) {
        val = _val;
        next = NULL;
    }

    Node(int _val, Node* _next) {
        val = _val;
        next = _next;
    }
};
*/

class Solution {
public:
    Node* insert(Node* head, int insertVal) {
        auto node = new Node(insertVal);
        if (!head) {
            node->next = node;
            return node;
        }
        auto pre = head, cur = head->next, max = head;
        do {
            if (pre->val <= insertVal && cur->val >= insertVal) {
                // 插入到这里就可以
                pre->next = node, node->next = cur;
                return head;
            }
            // 顺便找 max 
            if (pre->val >= max->val)
                max = pre;
            pre = cur, cur = cur->next;
        } while (pre != head);  // ATTENTION 判断条件为 head

        // 没有出现夹在中间的情况 需要放在 max 后
        node->next = max->next, max->next = node;
        return head;
    }
};
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

### 结合递归

> [!NOTE] **[LeetCode 430. 扁平化多级双向链表](https://leetcode.cn/problems/flatten-a-multilevel-doubly-linked-list/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* prev;
    Node* next;
    Node* child;
};
*/

class Solution {
public:
    Node* flatten(Node* head) {
        auto res = dfs(head);
        return res[0];
    }

    vector<Node*> dfs(Node* head) {
        if (!head) return {NULL, NULL};
        auto cur = head, tail = head;
        while (cur) {
            tail = cur;
            if (cur->child) {
                auto t = dfs(cur->child);
                cur->child = NULL;
                t[1]->next = cur->next;
                if (cur->next) cur->next->prev = t[1];
                cur->next = t[0];
                t[0]->prev = cur;
                cur = t[1]->next;
                tail = t[1];
            } else {
                cur = cur->next;
            }
        }
        return {head, tail};
    }
};
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *