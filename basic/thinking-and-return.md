## 习题

> [!NOTE] **[AcWing 1412. 邮政货车](https://www.acwing.com/problem/content/1414/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 也可以插头dp

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include <bits/stdc++.h>
using namespace std;

// 分析每一种形态状态
//  完整课 https://www.acwing.com/video/2305/
//  本题 https://www.acwing.com/video/3342/

const int N = 1010, M = 1010;

int n;
int w[6][6] = {
    {1, 0, 1, 1, 0, 0},
    {0, 1, 0, 0, 1, 0},
    {0, 1, 0, 0, 1, 0},
    {0, 1, 0, 0, 1, 0},
    {1, 0, 1, 1, 0, 1},
    {0, 0, 0, 0, 1, 0},
};
int f[N][6][M];

void add(int a[], int b[]) {
    for (int i = 0, t = 0; i < M; ++ i ) {
        t += a[i] + b[i];
        a[i] = t % 10;
        t /= 10;
    }
}

int main() {
    cin >> n;
    f[1][1][0] = f[1][4][0] = 1;
    for (int i = 2; i < n; ++ i )
        for (int j = 0; j < 6; ++ j )
            for (int k = 0; k < 6; ++ k )
                if (w[k][j])
                    add(f[i][j], f[i - 1][k]);
    int res[M] = {0};
    add(res, f[n - 1][0]), add(res, f[n - 1][4]);
    add(res, res);
    
    int k = M - 1;
    while (k > 0 && !res[k])
        k -- ;
    for (int i = k; i >= 0; -- i )
        cout << res[i];
    cout << endl;
    
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