> [!TIP] **【区间赋值】==> ODT树**
>
> <details>
> <summary>标准模版</summary>
>
>
> ```cpp
> struct Node_t {
>     int l, r;
>     mutable int v;
>     inline bool operator<(const Node_t & o) const {
>         return l < o.l;     // 按 l 升序排列
>     }
> };
> set<Node_t> odt;
> 
> auto split(int x) {
>     auto it = odt.lower_bound({x, 0, 0});  // 找到大于等于x的第一个
>     if (it != odt.end() && it->l == x)
>         return it;
>     // 否则x一定被前一段包含，向前移找到该段
>     it--;
>     auto [l, r, v] = *it;
> 
>     // 其他操作
>     // ...
> 
>     odt.erase(it);
>     odt.insert({l, x - 1, v});
>     return odt.insert({x, r, v}).first;  // ATTENTION 返回迭代器
> }
> 
> void merge(set<Node_t>::iterator it) {
>     if (it == odt.end() || it == odt.begin())
>         return;
>     auto lit = prev(it);
>     auto [ll, lr, lv] = *lit;
>     auto [rl, rr, rv] = *it;
>     if (lv == rv) {
>         odt.erase(lit), odt.erase(it), odt.insert({ll, rr, lv});
>         // ... 其他操作
>     }
> }
> void assign(int l, int r, int v) {
>     auto itr = split(r + 1), itl = split(l);  // 顺序不能颠倒
> 
>     /* 假定需要其他操作（如有遍历的需要）
>     for (auto it = itl; it != itr; ++it) {
>         auto [l, r, _] = *it;
>         ...
>     }
>     */
> 
>     // 清除一系列节点
>     odt.erase(itl, itr);
>     odt.insert({l, r, v});
> 
>     // 维护区间 【视情况而定】
>     merge(odt.lower_bound({l, 0, 0})), merge(itr);
> }
> ```
>
> </details>

## 名称简介

老司机树，ODT(Old Driver Tree)，又名珂朵莉树（Chtholly Tree)。起源自 [CF896C](https://codeforces.com/problemset/problem/896/C)。

## 核心思想

把值相同的区间合并成一个结点保存在 set 里面。

## 用处

如果要保证复杂度正确，必须保证数据随机。

详见 [Codeforces 上关于珂朵莉树的复杂度的证明](http://codeforces.com/blog/entry/56135?#comment-398940)。

更详细的严格证明见 [珂朵莉树的复杂度分析](https://zhuanlan.zhihu.com/p/102786071)。

对于 add，assign 和 sum 操作，用 set 实现的珂朵莉树的复杂度为 $O(n \log \log n)$，而用链表实现的复杂度为 $O(n \log n)$。

## 正文

首先，结点的保存方式：

```cpp
struct Node_t {
    int l, r;
    mutable int v;
    inline bool operator<(const Node_t & o) const {
        return l < o.l;     // 按 l 升序排列
    }
};
```

其中，`int v` 是自由指定的附加数据。

> [!NOTE] **`mutable` 关键字的含义是什么？**
> 
> `mutable` 的意思是“可变的”，让我们可以在后面的操作中修改 `v` 的值。在 C++ 中，mutable 是为了突破 const 的限制而设置的。被 mutable 修饰的变量（mutable 只能用于修饰类中的非静态数据成员），将永远处于可变的状态，即使在一个 const 函数中。
> 
> 这意味着，我们可以直接修改已经插入 `set` 的元素的 `v` 值，而不用将该元素取出后重新加入 `set`。

然后，我们定义一个 `set<Node_t> odt;` 来维护这些结点。为简化代码，可以 `typedef set<Node_t>::iterator iter`，当然在题目支持 C++11 时也可以使用 `auto`。

### split

`split` 是最核心的操作之一，它用于将原本包含点 $x$ 的区间（设为 $[l, r]$）分裂为两个区间 $[l, x)$ 和 $[x, r]$ 并返回指向后者的迭代器。
参考代码如下：

```cpp
auto split(int x) {
    auto it = odt.lower_bound({x, 0, 0});  // 找到大于等于x的第一个
    if (it != odt.end() && it->l == x)
        return it;
    // 否则x一定被前一段包含，向前移找到该段
    it--;
    auto [l, r, v] = *it;

    // 其他操作
    // ...

    odt.erase(it);
    odt.insert({l, x - 1, v});
    return odt.insert({x, r, v}).first;  // ATTENTION 返回迭代器
}
```

这个玩意有什么用呢？
任何对于 $[l,r]$ 的区间操作，都可以转换成 set 上 $[split(l),split(r + 1))$ 的操作。

### assign

另外一个重要的操作 `assign` 用于对一段区间进行赋值。

对于 ODT 来说，区间操作只有这个比较特殊，也是保证复杂度的关键。

如果 ODT 里全是长度为 $1$ 的区间，就成了暴力，但是有了 `assign`，可以使 ODT 的大小下降。

参考代码如下：

```cpp
void merge(set<Node_t>::iterator it) {
    if (it == odt.end() || it == odt.begin())
        return;
    auto lit = prev(it);
    auto [ll, lr, lv] = *lit;
    auto [rl, rr, rv] = *it;
    if (lv == rv) {
        odt.erase(lit), odt.erase(it), odt.insert({ll, rr, lv});
        // ... 其他操作
    }
}
void assign(int l, int r, int v) {
    auto itr = split(r + 1), itl = split(l);  // 顺序不能颠倒

    /* 假定需要其他操作（如有遍历的需要）
    for (auto it = itl; it != itr; ++it) {
        auto [l, r, _] = *it;
        ...
    }
    */

    // 清除一系列节点
    odt.erase(itl, itr);
    odt.insert({l, r, v});

    // 维护区间 【视情况而定】
    merge(odt.lower_bound({l, 0, 0})), merge(itr);
}
```

### 其他操作

套模板就好了，参考代码如下：

```cpp
void performance(int l, int r) {
    auto itr = split(r + 1), itl = split(l);
    for (; itl != itr; ++itl) {
        // Perform Operations here
    }
}
```

注：**珂朵莉树在进行求取区间左右端点操作时，必须先 split 右端点，再 split 左端点**。若先 split 左端点，返回的迭代器可能在 split 右端点的时候失效，可能会导致 RE。

## 习题

- [「SCOI2010」序列操作](https://www.luogu.com.cn/problem/P2572)
- [「SHOI2015」脑洞治疗仪](https://loj.ac/problem/2037)
- [「Luogu 2787」理理思维](https://www.luogu.com.cn/problem/P2787)
- [「Luogu 4979」矿洞：坍塌](https://www.luogu.com.cn/problem/P4979)

> [!NOTE] **[LeetCode 2213. 由单个字符重复的最长子字符串](https://leetcode-cn.com/problems/longest-substring-of-one-repeating-character/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 模拟都是细节
> 
> 本题有其他做法: 珂朵莉树 TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 模拟**

```cpp
class Solution {
public:
    using PII = pair<int, int>;
    const static int N = 1e5 + 10;
    
    vector<int> longestRepeating(string s, string queryCharacters, vector<int>& queryIndices) {
        set<PII> S;
        multiset<int> MS;
        int n = s.size();
        for (int i = 0; i < n; ++ i ) {
            int j = i + 1;
            while (j < n && s[j] == s[i])
                j ++ ;
            S.insert({i, j - 1});
            MS.insert(j - i);
            i = j - 1;
        }
        
        vector<int> res;
        int m = queryCharacters.size();
        for (int i = 0; i < m; ++ i ) {
            char c = queryCharacters[i];
            int id = queryIndices[i];
            if (s[id] == c) {
                res.push_back(*MS.rbegin());
                continue;
            }
            
            auto [l, r] = *prev(S.lower_bound({id, INT_MAX}));   // ATTENTION
            // 先删后加 因为可能删的和加的是同一个区间
            S.erase(S.find({l, r}));
            MS.erase(MS.find(r - l + 1));
            
            if (l < id) {
                S.insert({l, id - 1});
                MS.insert(id - l);
            }
            if (id < r) {
                S.insert({id + 1, r});
                MS.insert(r - id);
            }
            
            int nl = id, nr = id;
            if (id + 1 < n && s[id + 1] == c) {
                auto [pl, pr] = *S.lower_bound({id, INT_MAX});
                nr = pr;
                S.erase(S.find({pl, pr}));
                MS.erase(MS.find(pr - pl + 1));
            }
            if (id - 1 >= 0 && s[id - 1] == c) {
                auto [pl, pr] = *prev(S.lower_bound({id, INT_MIN}));
                nl = pl;
                S.erase(S.find({pl, pr}));
                MS.erase(MS.find(pr - pl + 1));
            }
            
            s[id] = c;
            S.insert({nl, nr});
            MS.insert(nr - nl + 1);
            res.push_back(*MS.rbegin());
        }
        // cout << endl;
        return res;
    }
};
```


##### **C++ 珂朵莉树**

```cpp
class Solution {
public:
    int n;
    multiset<int> S;    // 记录所有长度

    struct Node_t {
        int l, r;
        mutable int v;
        inline bool operator<(const Node_t & o) const {
            return l < o.l;     // 按 l 升序排列
        }
    };
    set<Node_t> odt;
    auto split(int x) {
        auto it = odt.lower_bound({x, 0, 0}); // 找到大于等于x的第一个
        if (it != odt.end() && it->l == x) return it;
        // 否则x一定被前一段包含，向前移找到该段
        it -- ;
        auto [l, r, v] = *it;
        
        // 本题特殊处理
        S.erase(S.find(r - l + 1));
        S.insert(x - l), S.insert(r - x + 1);   // ATTENTION 是 [x, r] 而非 [x+1, r]

        odt.erase(it);
        odt.insert({l, x - 1, v});          
        return odt.insert({x, r, v}).first;     // ATTENTION 返回迭代器
    }
    void merge(set<Node_t>::iterator it) {
        if (it == odt.end() || it == odt.begin())
            return;
        auto lit = prev(it);
        auto [ll, lr, lv] = *lit;
        auto [rl, rr, rv] = *it;
        if (lv == rv) {
            odt.erase(lit), odt.erase(it), odt.insert({ll, rr, lv});
            S.erase(S.find(lr - ll + 1)), S.erase(S.find(rr - rl + 1)), S.insert(rr - ll + 1);
        }
    }
    void assign(int l, int r, int v) {
        auto itr = split(r + 1), itl = split(l);    // 顺序不能颠倒

        // 对于本题需要干掉相关的数据 所以遍历一遍
        S.insert(r - l + 1);
        for (auto it = itl; it != itr; ++ it ) {
            auto [l, r, _] = *it;
            S.erase(S.find(r - l + 1));
        }

        // 清除一系列节点
        odt.erase(itl, itr);
        odt.insert({l, r, v});

        // 维护区间
        merge(odt.lower_bound({l, 0, 0})), merge(itr);
    }

    vector<int> longestRepeating(string s, string queryCharacters, vector<int>& queryIndices) {
        this->n = s.size();

        // build odt
        for (int i = 0; i < n; ++ i ) {
            int j = i;
            while (j < n && s[j] == s[i])
                j ++ ;
            odt.insert({i, j - 1, s[i]});
            S.insert(j - i);
            i = j - 1;
        }

        vector<int> res;
        int m = queryCharacters.size();
        for (int i = 0; i < m; ++ i ) {
            char c = queryCharacters[i];
            int idx = queryIndices[i];
            if (s[idx] == c) {
                res.push_back(*S.rbegin());
                continue;
            }
            s[idx] = c;
            assign(idx, idx, c);
            res.push_back(*S.rbegin());
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

> [!NOTE] **[LeetCode 2276. 统计区间中的整数数目](https://leetcode.cn/problems/count-integers-in-intervals/)**
> 
> 题意: 
> 
> 每次区间赋值 1 ，多次求总共多少个 1

> [!TIP] **思路**
> 
> 显然珂朵莉树，注意初始化即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class CountIntervals {
public:
    const static int INF = 2e9;
    
    int tot;
    struct Node_t {
        int l, r;
        mutable int v;
        inline bool operator<(const Node_t & o) const {
            return l < o.l;
        }
    };
    set<Node_t> odt;
    auto split(int x) {
        auto it = odt.lower_bound({x, 0, 0});
        if (it != odt.end() && it->l == x)
            return it;
        it -- ;
        auto [l, r, v] = *it;
        // ...
        odt.erase(it);
        odt.insert({l, x - 1, v});
        return odt.insert({x, r, v}).first;
    }
    void merge(set<Node_t>::iterator it) {
        if (it == odt.end() || it == odt.begin())
            return;
        auto lit = prev(it);
        auto [ll, lr, lv] = *lit;
        auto [rl, rr, rv] = *it;
        if (lv == rv) {
            odt.erase(lit), odt.erase(it), odt.insert({ll, rr, lv});
            // ...
        }
    }
    void assign(int l, int r, int v) {
        auto itr = split(r + 1), itl = split(l);
        
        // ... start
        for (auto it = itl; it != itr; ++ it ) {
            auto [tl, tr, tv] = *it;
            if (tv)
                tot -= tr - tl + 1;
        }
        tot += r - l + 1;
        // ... end
        
        odt.erase(itl, itr);
        odt.insert({l, r, v});
        merge(odt.lower_bound({l, 0, 0})), merge(itr);
    }
    
    CountIntervals() {
        tot = 0;
        odt.clear();
        odt.insert({-INF, INF, 0});
    }
    
    void add(int left, int right) {
        assign(left, right, 1);
    }
    
    int count() {
        return tot;
    }
};

/**
 * Your CountIntervals object will be instantiated and called as such:
 * CountIntervals* obj = new CountIntervals();
 * obj->add(left,right);
 * int param_2 = obj->count();
 */
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *