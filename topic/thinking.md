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

> [!NOTE] **[LeetCode 3035. 回文字符串的最大数量](https://leetcode.cn/problems/maximum-palindromes-after-operations/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 很巧妙的思维题 重在理清思路

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 题目本质就是 在长度不变的情况下，所有字符串的字符可以任意swap
    // 则 只与所有字符的计数 以及字符串长度有关

    int maxPalindromesAfterOperations(vector<string>& words) {
        // 考虑 先把左右的字母填了 最后在往正中间填入字母
        int tot = 0, mask = 0;
        for (auto & w : words) {
            tot += w.size();
            for (auto c : w)
                mask ^= 1 << (c - 'a');
        }
        // 总字符长度为tot 奇数有even位
        int even = __builtin_popcount(mask);
        // ATTENTION 先把奇数位都单独拎出来 剩下都是偶数个数【重要的细节推理 => 这个一定是和字符串长度一一匹配的】
        tot -= even;
        
        // 实际只会用到长度 [从短到长逐个填充]
        sort(words.begin(), words.end(), [](const string & a, const string & b) {
            return a.size() < b.size();
        });
        
        int res = 0;
        for (auto & w : words) {
            tot -= (w.size() / 2) * 2;  // 对于奇数来说 会消除掉1
            // ATTENTION 由于是从短到长 如果当前位置无法填充 后面的也没有填充可能 => 填充思想trick
            if (tot < 0)
                break;
            res ++ ;
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

> [!NOTE] **[LeetCode 3020. 子集中元素的最大数量](https://leetcode.cn/problems/find-the-maximum-number-of-elements-in-subset/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 细节略多容易WA
> 
> 主要在于时间复杂度分析

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 如果是 2 的幂次，最多指数到 32
    // 则 按照每一个底数去做幂次统计... 然后枚举即可
    using LL = long long;
    
    int maximumLength(vector<int>& nums) {
        unordered_map<LL, int> h;
        for (auto x : nums)
            h[x] ++ ;
        
        int res = 0;
        sort(nums.begin(), nums.end());
        for (auto v : nums) {
            if (v == 1) {
                res = max(res, (h[1] % 2 ? h[1] : h[1] - 1));   // ATTENTION: k=0
                continue;
            }
            
            int t = 0;
            LL x = v;
            while (h.count(x) && h[x] > 1)
                t += 2, x *= x;
            t += (h.count(x) ? 1 : -1);
            res = max(res, t);
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

> [!NOTE] **[LeetCode 3213. 最小代价构造字符串](https://leetcode.cn/problems/construct-string-with-minimum-cost/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 容易想到字符串哈希 string hash
> 
> 但即使用了字符串哈希，上述做法仍然是 O(n^2) 的
> 
> 关键在于，枚举的子串 t 的长度，如果压根就不出现在 words 中，那么无需枚举这样的 j，或者说长度。
> 
> 注意到，**设 L 是 words 中所有字符串的长度之和，那么 words 中至多有 O(sqrt(L)) 个长度不同的字符串** => trick
> 
> (考虑长度和 1+2+3+⋯≤L)
> 
> 所以我们只需要枚举这 O(sqrt(L)) 个长度，而不是枚举 O(n) 个 j
> 
> [计算复杂度分析]

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // f[i] = 填充target前i个字母的最小cost
    // f[i] = f[j] + cost(j + 1, i); 问题在于这是O(n^2)的
    //  => 维护所有可行的j + string hash
    
    // ATTENTION:
    // 但即使用了字符串哈希，上述做法仍然是 O(n^2) 的
    // 关键在于，枚举的子串 t 的长度，如果压根就不出现在 words 中，那么无需枚举这样的 j，或者说长度。
    // 注意到，设 L 是 words 中所有字符串的长度之和，那么 words 中至多有 O(sqrt(L)) 个长度不同的字符串 => trick
    // （考虑长度和 1+2+3+⋯≤L）
    // 所以我们只需要枚举这 O(sqrt(L)) 个长度，而不是枚举 O(n) 个 j
    
    using ULL = unsigned long long;
    const static int N = 5e4 + 10, P = 131, INF = 0x3f3f3f3f;
    
    ULL h[N], p[N];
    void init(string s) {
        int n = s.size();
        h[0] = 0, p[0] = 1;
        for (int i = 1; i <= n; ++ i )
            h[i] = h[i - 1] * P + s[i - 1], p[i] = p[i - 1] * P;    // ATTENTION P
    }
    ULL get(int l, int r) {
        return h[r] - h[l - 1] * p[r - l + 1];
    }
    
    int f[N];
    map<int, unordered_map<ULL, int>> min_cost; // ATTENTION map<int, unordered_map<...>>
    
    int minimumCost(string target, vector<string>& words, vector<int>& costs) {
        init(target);
        
        {
            int m = words.size();
            for (int i = 0; i < words.size(); ++ i ) {
                auto & w = words[i];
                int len = w.size();
                ULL hash = 0;
                for (int i = 1; i <= len; ++ i )
                    hash = hash * P + w[i - 1];
                if (min_cost[len].count(hash))
                    min_cost[len][hash] = min(min_cost[len][hash], costs[i]);
                else
                    min_cost[len][hash] = costs[i];
            }
        }
        
        int n = target.size();
        memset(f, 0x3f, sizeof f);
        f[0] = 0;
        
        for (int i = 1; i <= n; ++ i ) {
            for (auto & [len, cs] : min_cost) {   // O(sqrt(L)), 注意引用
                if (len > i)
                    break;
                int last = i - len;
                ULL hash = get(last + 1, i);
                if (cs.count(hash))
                    f[i] = min(f[i], f[last] + cs[hash]);
            }
        }
        return f[n] >= INF / 2 ? -1 : f[n];
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

> [!NOTE] **[LeetCode 3234. 统计 1 显著的字符串的数量](https://leetcode.cn/problems/count-the-number-of-substrings-with-dominant-ones/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 时间复杂度证明
> 
> 以及 计算逻辑细节

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // 类似题目 考虑枚举右端点 计算合法左端点的数量
    //
    // 感觉像是 O(n^2) 的做法...
    // => 实际上 0 的数量不会超过 log(len)
    //  考虑枚举 0 的数量，总的时间复杂度为 O(n*log(n))
    
    int numberOfSubstrings(string s) {
        int n = s.size();
        int res = 0;
        vector<int> zeros;
        zeros.push_back(0); // 哨兵
        for (int i = 1; i <= n; ++ i ) {
            int t = s[i - 1] - '0';
            if (t)  // 当前位置为 1
                res += i - zeros.back();    // ATTENTION 1. 不包含任何0的情况
            else
                zeros.push_back(i);
            
            // ATTENTION 2. 包含至少 1 个 0 的情况
            //
            // m-j 用来优化复杂度 避免无意义的遍历 k*k <= num_of_ones = i-m
            int m = zeros.size();
            for (int j = m - 1; j > 0 /*ATTENTION > 0*/ && (m - j) <= (i - (m - 1)) / (m - j); -- j ) {
                // 该区间内 0,1 的数量
                int c0 = m - j;
                int c1 = (i - zeros[j] + 1/*总长度 ATTENTION: +1*/) - c0;
                
                // 在 zeros[j] 左侧还有一段1 数量为 tot = zeros[j]-zeros[j-1]-1
                // 问题在于这些1 有一部分是已经不得不消耗的 具体为 cost = c0*c0-c1
                //      剩下的 比较灵活 即为 tot - cost + 1
                //                  zeros[j]-zeros[j-1] -1 - cost +1
                res += max(0, zeros[j] - zeros[j - 1] - max(0, c0 * c0 - c1));
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