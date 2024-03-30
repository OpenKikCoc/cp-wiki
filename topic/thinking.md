# 习题

## 思维

> [!NOTE] **[LeetCode 2790. 长度递增组的最大数目](https://leetcode.cn/problems/maximum-number-of-groups-with-increasing-length/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 非常经典的模型
> 
> 贪心排序 构造思维

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 考虑 按照数量降序
    // 第i次需要把前i大的各自-1 减完之后有可能发生顺序变化
    // 显然不能用堆枚举 复杂度无法接受
    using LL = long long;

    vector<int> uls;

    int maxIncreasingGroups(vector<int>& usageLimits) {
        this->uls = usageLimits;
        sort(uls.begin(), uls.end());
        
        LL rest = 0, res = 0;
        for (auto x : uls) {
            // 直接加  因为不管是“构造不了”还是“构造完有余”都可以给后面用
            // ATTENTION 思维与证明 为什么这样一定可以构造
            // 以柱状图的形式思考
            // 每次新加一个数构造相当于基于原本的再追加一列【一定能够通过置换使得其可用】
            rest += x;
            if (rest >= res + 1) {
                rest -= res + 1;
                res ++ ;
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

> [!NOTE] **[LeetCode 2860. 让所有学生保持开心的分组方法数](https://leetcode.cn/problems/happy-students/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 有点思维

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    const static int N = 1e5 + 10;
    
    int countWays(vector<int>& nums) {
        int n = nums.size();
        sort(nums.begin(), nums.end());
        
        int res = 0;
        // 枚举不被选择的左边界
        for (int i = 0, max_choosed = -1; i < n; ++ i ) {
            int choosed = i;
            if (choosed < nums[i] && choosed > max_choosed)
                res ++ ;
            max_choosed = nums[i];
        }
        // 如果全部选中
        if (n > nums.back())
            res ++ ;
        
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

> [!NOTE] **[LeetCode 2862. 完全子集的最大元素和](https://leetcode.cn/problems/maximum-element-sum-of-a-complete-subset-of-indices/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 容易想到按照因子的奇偶性对所有下标进行解析
> 
> 随后得到 st 值
> 
> 核心点在于：并非使用 st 记录奇偶性，而是记录对应奇偶的乘积 => 思考

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
using LL = long long;

class Solution {
public:
    // 注意: 所谓的 '完全平方数' 是指下标，而非下标对应的值
    //  => 不会有重复值，则只能使用 1-1e4 范围内的元素，且集合需要保证质数的幂次总和为偶数
    //          后者必然符合: 质数大小不超过 25 个
    //
    // 还有个重要条件: 每对元素都需要能够两两完全平方 => 则一组内 所有元素的 st 必须完全一样
    // 
    // 【思考】ATTENTION: 不能比较二进制的 st，而应当计算乘积
    
    vector<int> ps = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};
    int cnt = 25;
    
    long long maximumSum(vector<int>& nums) {
        int n = nums.size();
        
        LL res = 0;
        unordered_map<int, LL> h;
        for (int i = 1; i <= n; ++ i ) {
            // int st = 0, x = i;
            int st = 1, x = i;
            for (int j = 0; j < 25; ++ j ) {
                int c = 0, p = ps[j];
                while (x % p == 0)
                    x /= p, c ^= 1;
                if (c)
                    // st ^= 1 << j;
                    st *= p;
            }
            if (x > 1) {
                // 错误思维:
                // 1. 只能自己成为一组的元素
                // res = max(res, (LL)nums[i - 1]);    // ATTENTION: WA1, 需要单独考虑一下
                // continue;
                st *= x;
            }
            // 2. 可选的，能够与其他元素组合的元素
            h[st] += nums[i - 1];
        }

        for (auto & [x, y] : h)
            res = max(res, y);
        
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

> [!NOTE] **[LeetCode 2939. 最大异或乘积](https://leetcode.cn/problems/maximum-xor-product/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 思维题 参考注释思路

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 52, MOD = 1e9 + 7;
    
    // 思考 n 限制了可以自由 0/1 的总位数 => 因为 a/b 位数可能大于 n
    //   如何使得结果最大? 显然需要让 a^x, b^x 各自最大
    // 考虑: 如果 a/b 的某个位都是 0/0 则有必要使得x取 1 让 a/b + 1<<y
    //      如果 a/b 的某个位分别是 1/0, 0/1 则只能有两种选择 => 二者分别 +/- 1<<y
    // 最坏情况下 50个位都需要 +/- 
    // ======> 贪心: 第一个位给其中一个数，剩下的位都给另一个数即可 => 需要求出 x
    
    int maximumXorProduct(long long a, long long b, int n) {
        vector<int> t;
        // ATTENTION 可以等于
        for (int i = 0; i < n; ++ i ) {
            int x = (a >> i) & 1, y = (b >> i) & 1;
            if (x == 0 && y == 0) {
                a += 1ll << i, b += 1ll << i;
            } else if (x == 1 && y == 1) {
                // do nothing
                continue;
            } else {
                t.push_back(i);     // ATTENTION 简化，不需要关心是谁转移到谁
                // ATTENTION 但是需要把 a, b 在此位消 0
                if (x)
                    a -= 1ll << i;  // ATTENTION use 1ll
                else
                    b -= 1ll << i;
            }
        }
        
        // 【核心在于分配策略】
        if (a == b) {
            // 对于所有不同的位 需要把最高位给 a 其余都给 b
            for (int i = t.size() - 1; i >= 0; -- i ) {
                if (i == t.size() - 1)
                    a += 1ll << t[i];
                else
                    b += 1ll << t[i];
            }
        } else {
            // 全给 a 或全给 b
            LL tot = 0;
            for (auto x : t)
                tot += 1ll << x;
            if (a < b)
                a += tot;
            else
                b += tot;
        }
        a %= MOD, b %= MOD; // ATTENTION
        // cout << " a = " << a << " b = " << b << endl;
        return a * b % MOD;
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

> [!NOTE] **[LeetCode 3012. 通过操作使数组长度最小](https://leetcode.cn/problems/minimize-length-of-array-using-operations/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 本质: 任意挑两个数，小的干掉大的并能够在末尾留下 [小的/0]
> 
>   1. 消耗较大的数，结果最后只剩下一堆最小的数
> 
>   2. 这堆最小的数内部消耗
> 
> 容易知道，如果最小的数只有 1 个，最终答案就是 1
> 
> 如果不止一个，需要探索能够构造一个比全局最小数更小的数
> 
> > ATTNTION 自己想的是相邻数字取模... 实际上可以直接对【最小的数字取模】...【思维】

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using PII = pair<int, int>;
    
    bool check(vector<PII> & t, int v) {
        for (int i = 0; i < t.size(); ++ i ) {
            // WRONG
            // int mod = t[i].first % t[i - 1].first;
            
            int mod = t[i].first % v;
            if (mod /*non-zero*/ && mod < v)
                return true;
        }
        return false;
    }
    
    int minimumArrayLength(vector<int>& nums) {
        unordered_map<int, int> h;
        for (auto x : nums)
            h[x] ++ ;
        
        vector<PII> t;
        for (auto [k, v] : h)
            t.push_back({k, v});
        sort(t.begin(), t.end());

        auto x = t[0].second;
        if (x == 1)
            return 1;
        else if (check(t, t[0].first))
            return 1;
        
        return (x + 1) / 2;
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

## 计算复杂度分析

> [!NOTE] **[LeetCode 2910. 合法分组的最少组数](https://leetcode.cn/problems/minimum-number-of-groups-to-create-a-valid-assignment/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 枚举的计算复杂度分析
> 
> > 设哈希表的大小为 m 哈希表中最小的 value 为 k
> > 
> > 由于所有 value 之和为 n 所以 $km \le n$
> > 
> > 而循环次数又至多为 km 所以时间复杂度为 $\mathcal{O}(n)$

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<int> xs;

    // 按照某个 sz 分组能否得到可行解
    // ATTENTION sz 是下界，可以出现个数为 sz+1 的情况
    bool check(int sz) {
        for (auto x : xs)
            if (x % sz > x / sz)    // ATTENTION 判断规则 (可以等 相当于数量为sz+1)
                return false;
        return true;
    }

    int minGroupsForValidAssignment(vector<int>& nums) {
        // 不关心值具体是啥 只关心出现次数
        unordered_map<int, int> h;
        for (auto x : nums)
            h[x] ++ ;
        
        int minv = 1e9;
        xs.clear();
        for (auto [k, v] : h)
            xs.push_back(v), minv = min(minv, v);
        
        // 已知多个 "次数" 如何分配能够得到可行组
        // 考虑枚举某个分组状况下(大小) 是否能构造出合理方案
        int p;
        for (p = minv; p >= 2; -- p )
            if (check(p))
                break;
        
        int res = 0;
        for (auto x : xs)
            res += (x + p) / (p + 1);   // ATTENTION 向上取整
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