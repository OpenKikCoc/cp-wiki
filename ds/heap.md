

堆是一棵树，其每个节点都有一个键值，且每个节点的键值都大于等于/小于等于其父亲的键值。

每个节点的键值都大于等于其父亲键值的堆叫做小根堆，否则叫做大根堆。[STL 中的 `priority_queue`](/lang/csl/container-adapter/#_13) 其实就是一个大根堆。

（小根）堆主要支持的操作有：插入一个数、查询最小值、删除最小值、合并两个堆、减小一个元素的值。

一些功能强大的堆（可并堆）还能（高效地）支持 merge 等操作。

一些功能更强大的堆还支持可持久化，也就是对任意历史版本进行查询或者操作，产生新的版本。

## 堆的分类

|         操作\\数据结构        |                                  配对堆                                  |      二叉堆     |      左偏树     |      二项堆     |    斐波那契堆    |
| :---------------------: | :-------------------------------------------------------------------: | :----------: | :----------: | :----------: | :---------: |
|        插入（insert）       |                                 $O(1)$                                |  $O(\log n)$ |  $O(\log n)$ |    $O(1)$    |    $O(1)$   |
|     查询最小值（find-min）     |                                 $O(1)$                                |    $O(1)$    |    $O(1)$    |  $O(\log n)$ |    $O(1)$   |
|    删除最小值（delete-min）    |                              $O(\log n)$                              |  $O(\log n)$ |  $O(\log n)$ |  $O(\log n)$ | $O(\log n)$ |
|        合并 (merge)       |                                 $O(1)$                                |    $O(n)$    |  $O(\log n)$ |  $O(\log n)$ |    $O(1)$   |
| 减小一个元素的值 (decrease-key) | $o(\log n)$（下界 $\Omega(\log \log n)$，上界 $O(2^{2\sqrt{\log \log n}})$) |  $O(\log n)$ |  $O(\log n)$ |  $O(\log n)$ |    $O(1)$   |
|         是否支持可持久化        |                                $\times$                               | $\checkmark$ | $\checkmark$ | $\checkmark$ |   $\times$  |

习惯上，不加限定提到“堆”时往往都指二叉堆。

> [!NOTE] **[AcWing 839. 模拟堆](https://www.acwing.com/problem/content/841/)**
> 
> 题意: TODO

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 100010;

int h[N], ph[N], hp[N], cnt;

void heap_swap(int a, int b) {
    swap(ph[hp[a]], ph[hp[b]]);
    swap(hp[a], hp[b]);
    swap(h[a], h[b]);
}

void down(int u) {
    int t = u;
    if (u * 2 <= cnt && h[u * 2] < h[t]) t = u * 2;
    if (u * 2 + 1 <= cnt && h[u * 2 + 1] < h[t]) t = u * 2 + 1;
    if (u != t) {
        heap_swap(u, t);
        down(t);
    }
}

void up(int u) {
    while (u / 2 && h[u] < h[u / 2]) {
        heap_swap(u, u / 2);
        u >>= 1;
    }
}

int main() {
    int n, m = 0;
    cin >> n;
    while (n -- ) {
        char op[5];
        int k, x;
        cin >> op;
        if (op[0] == 'I') {
            cin >> x;
            cnt ++ ;
            m ++ ;
            ph[m] = cnt, hp[cnt] = m;
            h[cnt] = x;
            up(cnt);
        } else if (op[0] == 'P' && op[1] == 'M') cout << h[1] << endl;
        else if (op[0] == 'D' && op[1] == 'M') {
            heap_swap(1, cnt);
            cnt -- ;
            down(1);
        } else if (op[0] == 'D') {
            cin >> k;
            k = ph[k];
            heap_swap(k, cnt);
            cnt -- ;
            up(k);
            down(k);
        } else {
            cin >> k >> x;
            k = ph[k];
            h[k] = x;
            up(k);
            down(k);
        }
    }
    return 0;
}
```

##### **Python**

```python
```

<!-- tabs:end -->
</details>

<br>

* * *