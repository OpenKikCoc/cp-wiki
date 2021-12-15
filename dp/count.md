
## 习题

> [!NOTE] **[AcWing 900. 整数划分](https://www.acwing.com/problem/content/description/902/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 完全背包**

```cpp
/*
  完全背包解法

  状态表示：
  f[i][j]表示只从 1~i 中选，且总和等于 j 的方案数

  状态转移方程:
  f[i][j] = f[i - 1][j] + f[i][j - i];
*/
#include <algorithm>
#include <iostream>

using namespace std;

const int N = 1010, mod = 1e9 + 7;

int n;
int f[N];

int main() {
    cin >> n;

    f[0] = 1;
    for (int i = 1; i <= n; i++)
        for (int j = i; j <= n; j++) f[j] = (f[j] + f[j - i]) % mod;

    cout << f[n] << endl;

    return 0;
}

```


##### **C++ 其他定义**

```cpp
/*
  其他算法
  状态表示：
  f[i][j] 表示总和为 i ，总个数为 j 的方案数

  状态转移方程：
  f[i][j] = f[i - 1][j - 1] + f[i - j][j];
*/
#include <algorithm>
#include <iostream>

using namespace std;

const int N = 1010, mod = 1e9 + 7;

int n;
int f[N][N];

int main() {
    cin >> n;

    f[1][1] = 1;
    for (int i = 2; i <= n; i++)
        for (int j = 1; j <= i; j++)
            f[i][j] = (f[i - 1][j - 1] + f[i - j][j]) % mod;

    int res = 0;
    for (int i = 1; i <= n; i++) res = (res + f[n][i]) % mod;

    cout << res << endl;

    return 0;
}
```


##### **Python**

```python
# 方法1：背包做法
# 整数n是背包容量n，有n个物品，物品的体积分别是1-n，每个物品可以用无数次，求恰好装满背包的方案数（完全背包问题）
# 状态表示：f[i,j]：从1-i中选，并且体积恰好是j的选法数量；属性：数量
# 状态转移：根据第i个物品 选几个来划分：选0个，，，选k个...
# f[i-1,j], f[i-1,j-i],f[i-1,j-2i],...
# 状态数量是n*n, 转移数量是n，所以时间复杂度是n*n*n

# 精益求精，按照完全背包问题的方法进行优化：
f[i][j] = f[i - 1][j] + f[i - 1][j - i] + f[i - 1][j - 2
i]+..+f[i - 1][j - i * s]
f[i][j - 1] = f[i - 1][j - 1] + f[i - 1][j - 2
i]+..+f[i - 1][j - i * s]
# ==> 最后优化出来的状态转移方程是：
f[i][j] = f[i - 1][j] + f[i][j - 1]

# 然后最后再进行空间优化，体积从小到大循环就可以去掉一维。
if __name__ == '__main__':
    N = 1010
    n = int(input())
    f = [0] * N
    mod = int(1e9 + 7)
    f[0] = 1
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            f[j] = (f[j] + f[j - i]) % mod
    print(f[n])

#	方法2:脑筋急转弯方案!!!很难想!!!

# 状态表示：f[i,j]集合表示：所有总和是i,并且恰好表示成j个数的和的方案；属性：数量
# 状态转移：以集合里的最小值是否大于1划分：
# 1） 最小值是1；f[i,j]=f[i-1,j-1] ：减去数字1的方案数，那就是总和i-1,个数j-1
# 2） 最小值大于1；集合里的每个数都减1，那就是总和i-1*j,个数还是j;
# 表达式f[i,j]=f[i-1,j-1]+f[i-1,j]
# 最后的答案 需要枚举一遍：ans=f[n,1]+f[n,2]+...+f[n,n]
if __name__ == '__main__':
    N = 1010
    n = int(input())
    f = [[0] * N for _ in range(N)]
    mod = int(1e9 + 7)

    f[1][1] = 1
    for i in range(2, n + 1):
        for j in range(1, i + 1):
            f[i][j] = (f[i - 1][j - 1] + f[i - j][j]) % mod
    res = 0
    for i in range(1, n + 1):
        res = (res + f[n][i]) % mod
    print(res)
```

<!-- tabs:end -->
</details>

<br>

* * *