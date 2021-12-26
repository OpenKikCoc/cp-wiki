## 习题

> [!NOTE] **[AcWing 95. 费解的开关](https://www.acwing.com/problem/content/97/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 二进制枚举 + 贪心 + 递推

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
#include<bits/stdc++.h>

using namespace std;

const int N = 6;

char g[N][N], bg[N][N];

int dx[5] = {-1, 0, 1, 0, 0}, dy[5] = {0, 1, 0, -1, 0};

void turn(int x, int y) {
    for (int i = 0; i < 5; ++ i ) {
        int nx = x + dx[i], ny = y + dy[i];
        if (nx < 0 || nx >= 5 || ny < 0 || ny >= 5) continue;
        g[nx][ny] ^= 1;
    }
}

int main() {
    int T;
    scanf("%d", &T);
    while (T -- ) {
        for (int i = 0; i < 5; ++ i ) scanf("%s", bg[i]);
        int res = 10;
        // 枚举第一行的所有操作方案 2^5
        for (int op = 0; op < 32; ++ op ) {
            int cnt = 0;
            memcpy(g, bg, sizeof g);
            for (int i = 0; i < 5; ++ i )
                if (op >> i & 1) {
                    turn(0, i);
                    ++ cnt;
                }
            // 递推1~4行开关的状态
            for (int i = 0; i < 4; ++ i )
                for (int j = 0; j < 5; ++ j )
                    if (g[i][j] == '0') {
                        turn(i + 1, j);
                        ++ cnt;
                    }
            bool success = true;
            for (int i = 0; i < 5; ++ i )
                if (g[4][i] == '0')
                    success = false;
            if (success && res > cnt) res = cnt;
        }
        if (res > 6) res = -1;
        printf("%d\n", res);
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
