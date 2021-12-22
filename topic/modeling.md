
## 习题

> [!NOTE] **[AcWing 1361. 三值序列排序](https://www.acwing.com/problem/content/1363/)**
> 
> 题意: TODO

> [!TIP] **思路**
>
> 归并统计逆序对本质是每次交换【相邻】元素的冒泡排序
> 
> 本题可以直接交换任意位置的两个元素
> 
> 考虑根据原数组与目标数组间的关系 建图
> 
> $$
> [1...1][2...2][3...3] \\
> [21231][12123][31231]
> $$
> 
> 必然有一些没有变动 有一些交换
> 
> x占了y的位置 则连一条从x->y的边 出边=入边 最终必然形成一个欧拉图
>
> - 交换的第一种结果：将一个三数环 (3) 拆开
> 
> - 交换的第二种结果：将两个环 (1 + 2) 合并
>
> 显然 交换次数必然 >= n - m (m为环数)
> 
> 目标为找到最多有多少个环
>
> 先统计各种边的数量 再找到最多多少个环
> 
> > more at https://www.acwing.com/video/2108/

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 1010;

int n;
int a[N], b[N];

int main() {
    cin >> n;
    int s[4] = {0};
    for (int i = 0; i < n; ++ i ) {
        cin >> a[i];
        s[a[i]] ++ ;
    }
    
    for (int i = 1, k = 0; i <= 3; ++ i )
        for (int j = 0; j < s[i]; ++ j )
            b[k ++ ] = i;
    
    int e[4][4] = {0};
    for (int i = 0; i < n; ++ i )
        e[a[i]][b[i]] ++ ;
    
    int m = 0;
    // 自环
    for (int i = 1; i <= 3; ++ i ) m += e[i][i];
    // 小环
    for (int i = 1; i <= 3; ++ i )
        for (int j = i + 1; j <= 3; ++ j ) {
            int t = min(e[i][j], e[j][i]);
            m += t;
            e[i][j] -= t, e[j][i] -= t;
        }
    // 最终的大环 这里任选选择了 1->2 和 2->1
    // 其中必然有一个数值为 0
    m += e[1][2] + e[2][1];
    cout << n - m << endl;
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