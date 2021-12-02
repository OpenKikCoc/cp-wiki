`双向同时搜索` 和 `Meet in the middle`

## 双向同时搜索

双向同时搜索的基本思路是从状态图上的起点和终点同时开始进行 `广搜` 或 `深搜`。如果发现搜索的两端相遇了，那么可以认为是获得了可行解。

双向广搜的步骤：

```text
将开始结点和目标结点加入队列 q
标记开始结点为 1
标记目标结点为 2
while (队列 q 不为空) {
    从 q.front() 扩展出新的 s 个结点
  
    如果 新扩展出的结点已经被其他数字标记过
        那么 表示搜索的两端碰撞
        那么 循环结束
  
    如果 新的 s 个结点是从开始结点扩展来的
        那么 将这个 s 个结点标记为 1 并且入队 q 
    
    如果 新的 s 个结点是从目标结点扩展来的
        那么 将这个 s 个结点标记为 2 并且入队 q
}
```

AcWing 双向广搜写法:

[AcWing 190. 字串变换](https://www.acwing.com/problem/content/192/)

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
int n;
string a[N], b[N];

int extend(queue<string>& q, unordered_map<string, int>& da, unordered_map<string, int>& db, string a[], string b[]) {
    for (int k = 0, sk = q.size(); k < sk; k ++ ) {
        string t = q.front();
        q.pop();

        for (int i = 0; i < t.size(); i ++ )
            for (int j = 0; j < n; j ++ )
                if (t.substr(i, a[j].size()) == a[j]) {
                    string state = t.substr(0, i) + b[j] + t.substr(i + a[j].size());
                    if (da.count(state)) continue;
                    if (db.count(state)) return da[t] + 1 + db[state];
                    da[state] = da[t] + 1;
                    q.push(state);
                }
    }

    return 11;
}

int bfs(string A, string B) {
    queue<string> qa, qb;
    unordered_map<string, int> da, db;
    qa.push(A), da[A] = 0;
    qb.push(B), db[B] = 0;

    while (qa.size() && qb.size()) {
        int t;
        if (qa.size() <= qb.size()) t = extend(qa, da, db, a, b);
        else t= extend(qb, db, da, b, a);

        if (t <= 10) return t;
    }

    return 11;
}

int main() {
    string A, B;
    cin >> A >> B;
    while (cin >> a[n] >> b[n]) n ++ ;

    int step = bfs(A, B);
    if (step > 10) puts("NO ANSWER!");
    else printf("%d\n", step);

    return 0;
}
```

##### **Python**

```python
```

<!-- tabs:end -->
</details>

<br>

## Meet in the middle

> [!WARNING]
> 
> 本节要介绍的不是 [**二分搜索**](../basic/binary.md)（二分搜索的另外一个译名为“折半搜索”）。

Meet in the middle 算法没有正式译名，常见的翻译为「折半搜索」、「双向搜索」或「中途相遇」。**它适用于输入数据较小，但还没小到能直接使用暴力搜索的情况。**

主要思想是将整个搜索过程分成两半，分别搜索，最后将两半的结果合并。

暴力搜索的复杂度往往是指数级的，而改用 meet in the middle 算法后复杂度的指数可以减半，即让复杂度从 $O(a^b)$ 降到 $O(a^{b/2})$。

> [!NOTE] **例题 [「USACO09NOV」灯 Lights](https://www.luogu.com.cn/problem/P2962)**
> 
> 有 $n$ 盏灯，每盏灯与若干盏灯相连，每盏灯上都有一个开关，如果按下一盏灯上的开关，这盏灯以及与之相连的所有灯的开关状态都会改变。一开始所有灯都是关着的，你需要将所有灯打开，求最小的按开关次数。
> $1\le n\le 35$。

> [!TIP] **解题思路**
> 
> 如果这道题暴力 DFS 找开关灯的状态，时间复杂度就是 $O(2^{n})$, 显然超时。
> 
> 不过，如果我们用 meet in middle 的话，时间复杂度可以优化至 $O(n2^{n/2})$。
> 
> meet in middle 就是让我们先找一半的状态，也就是找出只使用编号为 $1$ 到 $\mathrm{mid}$ 的开关能够到达的状态，再找出只使用另一半开关能到达的状态。
> 
> 如果前半段和后半段开启的灯互补，将这两段合并起来就得到了一种将所有灯打开的方案。
> 
> 具体实现时，可以把前半段的状态以及达到每种状态的最少按开关次数存储在 map 里面，搜索后半段时，每搜出一种方案，就把它与互补的第一段方案合并来更新答案。

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
```

##### **Python**

```python
```

<!-- tabs:end -->
</details>

<br>

## 习题

TODO

## 参考资料与注释

- [What is meet in the middle algorithm w.r.t. competitive programming? - Quora](https://www.quora.com/What-is-meet-in-the-middle-algorithm-w-r-t-competitive-programming)
- [Meet in the Middle Algorithm - YouTube](https://www.youtube.com/watch?v=57SUNQL4JFA)
