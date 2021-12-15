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