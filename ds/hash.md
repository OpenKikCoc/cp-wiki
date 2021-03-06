## 哈希表

![](images/hashtable.svg)

哈希表是又称散列表，一种以 "key-value" 形式存储数据的数据结构。所谓以 "key-value" 形式存储数据，是指任意的键值 key 都唯一对应到内存中的某个位置。只需要输入查找的键值，就可以快速地找到其对应的 value。可以把哈希表理解为一种高级的数组，这种数组的下标可以是很大的整数，浮点数，字符串甚至结构体。

## 哈希函数

要让键值对应到内存中的位置，就要为键值计算索引，也就是计算这个数据应该放到哪里。这个根据键值计算索引的函数就叫做哈希函数，也称散列函数。举个例子，如果键值是一个人的身份证号码，哈希函数就可以是号码的后四位，当然也可以是号码的前四位。生活中常用的“手机尾号”也是一种哈希函数。在实际的应用中，键值可能是更复杂的东西，比如浮点数、字符串、结构体等，这时候就要根据具体情况设计合适的哈希函数。哈希函数应当易于计算，并且尽量使计算出来的索引均匀分布。

在 OI 中，最常见的情况应该是键值为整数的情况。当键值的范围比较小的时候，可以直接把键值作为数组的下标，但当键值的范围比较大，比如以 $10^9$ 范围内的整数作为键值的时候，就需要用到哈希表。一般把键值模一个较大的质数作为索引，也就是取 $f(x)=x \bmod M$ 作为哈希函数。另一种比较常见的情况是 key 为字符串的情况，在 OI 中，一般不直接把字符串作为键值，而是先算出字符串的哈希值，再把其哈希值作为键值插入到哈希表里。

能为 key 计算索引之后，我们就可以知道每个键值对应的值 value 应该放在哪里了。假设我们用数组 a 存放数据，哈希函数是 f，那键值对 `(key, value)` 就应该放在 `a[f(key)]` 上。不论键值是什么类型，范围有多大，`f(key)` 都是在可接受范围内的整数，可以作为数组的下标。

## 冲突

如果对于任意的键值，哈希函数计算出来的索引都不相同，那只用根据索引把 `(key, value)` 放到对应的位置就行了。但实际上，常常会出现两个不同的键值，他们用哈希函数计算出来的索引是相同的。这时候就需要一些方法来处理冲突。在 OI 中，最常用的方法是拉链法。

### 拉链法

拉链法也称开散列法（open hashing）。

拉链法是在每个存放数据的地方开一个链表，如果有多个键值索引到同一个地方，只用把他们都放到那个位置的链表里就行了。查询的时候需要把对应位置的链表整个扫一遍，对其中的每个数据比较其键值与查询的键值是否一致。如果索引的范围是 $1\ldots M$，哈希表的大小为 $N$，那么一次插入/查询需要进行期望 $O(\frac{N}{M})$ 次比较。

#### 实现

```cpp
// C++ Version
```

```python
# Python Version
```

提供一个封装过的模板，可以像 map 一样用，并且较短

```cpp
struct hash_map {  // 哈希表模板
    struct data {
        long long u;
        int v, nex;
    };                // 前向星结构
    data e[SZ << 1];  // SZ 是 const int 表示大小
    int h[SZ], cnt;
    int hash(long long u) { return u % SZ; }
    int& operator[](long long u) {
        int hu = hash(u);  // 获取头指针
        for (int i = h[hu]; i; i = e[i].nex)
            if (e[i].u == u) return e[i].v;
        return e[++cnt] = (data){u, -1, h[hu]}, h[hu] = cnt, e[cnt].v;
    }
    hash_map() {
        cnt = 0;
        memset(h, 0, sizeof(h));
    }
};
```

在这里，hash 函数是针对键值的类型设计的，并且返回一个链表头指针用于查询。在这个模板中我们写了一个键值对类型为 `(long long, int)` 的 hash 表，并且在查询不存在的键值时返回 -1。函数 `hash_map()` 用于在定义时初始化。

### 闭散列法

闭散列方法把所有记录直接存储在散列表中，如果发生冲突则根据某种方式继续进行探查。

比如线性探查法：如果在 `d` 处发生冲突，就依次检查 `d + 1`，`d + 2`……

#### 实现

```cpp
const int N = 360007;  // N 是最大可以存储的元素数量
class Hash {
private:
    int keys[N];
    int values[N];

public:
    Hash() { memset(values, 0, sizeof(values)); }
    int& operator[](int n) {
        // 返回一个指向对应 Hash[Key] 的引用
        // 修改成不为 0 的值 0 时候视为空
        int idx = (n % N + N) % N, cnt = 1;
        while (keys[idx] != n && values[idx] != 0) {
            idx = (idx + cnt * cnt) % N;
            cnt += 1;
        }
        keys[idx] = n;
        return values[idx];
    }
};
```

## 例题

[「JLOI2011」不重复数字](https://www.luogu.com.cn/problem/P4305)

## 习题

> [!NOTE] **[AcWing 840. 模拟散列表](https://www.acwing.com/problem/content/842/)**
> 
> 题意: TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 开放寻址法**

```cpp
// 开放寻址法
#include <cstring>
#include <iostream>

using namespace std;

const int N = 200003, null = 0x3f3f3f3f;

int h[N];

int find(int x) {
    int t = (x % N + N) % N;
    while (h[t] != null && h[t] != x) {
        t++;
        if (t == N) t = 0;
    }
    return t;
}

int main() {
    memset(h, 0x3f, sizeof h);

    int n;
    scanf("%d", &n);

    while (n--) {
        char op[2];
        int x;
        scanf("%s%d", op, &x);
        if (*op == 'I')
            h[find(x)] = x;
        else {
            if (h[find(x)] == null)
                puts("No");
            else
                puts("Yes");
        }
    }

    return 0;
}
```

##### **C++ 拉链法**

```cpp
// 拉链法
#include <cstring>
#include <iostream>

using namespace std;

const int N = 100003;

int h[N], e[N], ne[N], idx;

void insert(int x) {
    int k = (x % N + N) % N;
    e[idx] = x;
    ne[idx] = h[k];
    h[k] = idx++;
}

bool find(int x) {
    int k = (x % N + N) % N;
    for (int i = h[k]; i != -1; i = ne[i])
        if (e[i] == x) return true;

    return false;
}

int main() {
    int n;
    scanf("%d", &n);

    memset(h, -1, sizeof h);

    while (n--) {
        char op[2];
        int x;
        scanf("%s%d", op, &x);

        if (*op == 'I')
            insert(x);
        else {
            if (find(x))
                puts("Yes");
            else
                puts("No");
        }
    }

    return 0;
}
```
##### **Python**

```python
def insert(x):
    global idx
    k = x % N
    idx += 1
    ev[idx] = x
    ne[idx] = h[k]
    h[k] = idx


def query(x):
    k = x % N
    i = h[k]
    while i != -1:
        j = ev[i]
        if j == x:
            return True
        i = ne[i]
    return False


if __name__ == '__main__':
    N = 100010
    h = [-1] * N
    ev = [0] * N
    ne = [0] * N
    idx = 0

    n = int(input())
    for _ in range(n):
        op = input().split()
        if op[0] == 'I':
            insert(int(op[1]))
        else:
            if query(int(op[1])):
                print("Yes")
            else:
                print("No")
```

<!-- tabs:end -->
</details>

<br>

* * *