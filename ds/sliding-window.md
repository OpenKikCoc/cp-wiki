## 习题

> [!NOTE] **[LeetCode 30. 串联所有单词的子串](https://leetcode-cn.com/problems/substring-with-concatenation-of-all-words/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<int> findSubstring(string s, vector<string>& words) {
        vector<int> res;
        if (words.empty()) return res;
        int n = s.size(), m = words.size(), w = words[0].size();
        unordered_map<string, int> tot;
        for (auto& word : words) tot[word] ++ ;

        for (int i = 0; i < w; i ++ ) {
            unordered_map<string, int> wd;
            int cnt = 0;
            for (int j = i; j + w <= n; j += w) {
                if (j >= i + m * w) {
                    auto word = s.substr(j - m * w, w);
                    wd[word] -- ;
                    if (wd[word] < tot[word]) cnt -- ;
                }
                auto word = s.substr(j, w);
                wd[word] ++ ;
                if (wd[word] <= tot[word]) cnt ++ ;
                if (cnt == m) res.push_back(j - (m - 1) * w);
            }
        }

        return res;
    }
};
```

##### **Python**

```python
#枚举所有起始位置，长度len(words[0])，0，w,2w,...; 1,w+1,w+2,...;w-1,2w-1,...
#这样枚举的好处：每一个单词会出现在某一区间，不存在跨区间的情况。
#每个区间看成一整体，问题变成：找到连续区间 恰好是我们给定的元素
#每次滑动窗口往前移动一位，后面也会往前走一位。（加一个新的区间，删除一个旧的区间）
#如何判断两个集合（哈希）是否相等？
#用一个变量cnt存滑动窗口里的集合 也在words集合里出现的集合的有效数。cnt是否和words的集合数是够一致，就可以判断是否满足题意。

#时间复杂度，每组：O((n/w)*w) w组，所以O(N)=O(NW)

class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        from collections import defaultdict
        res = []
        if not words:return res
        
        n = len(s); m = len(words); w = len(words[0])
        tot = collections.defaultdict(int)
        for i in words:
            tot[i] += 1

        for l in range(w):
            cnt = 0
            wd = collections.defaultdict(int)
            r = l
            while r + w <= n:
                #维护窗口大小，把左边往前缩紧一个（因为r也是每次遍历只加入一个）
                if r >= l + m * w:
                    word = s[r - m * w : r - (m - 1) * w]
                    wd[word] -= 1
                    if wd[word] < tot[word]:
                        cnt -= 1
                #加入当前处理的右指针指向的即将加入的单词
                word = s[r: r + w]
                wd[word] += 1
                if wd[word] <= tot[word]:
                    cnt += 1
                if cnt == m:
                    res.append(r - (m - 1) * w)
                r += w
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    string minWindow(string s, string t) {
        unordered_map<char, int> hs, ht;
        for (auto c: t) ht[c] ++ ;

        string res;
        int cnt = 0;
        for (int i = 0, j = 0; i < s.size(); i ++ ) {
            hs[s[i]] ++ ;
            if (hs[s[i]] <= ht[s[i]]) cnt ++ ;

            while (hs[s[j]] > ht[s[j]]) hs[s[j ++ ]] -- ;
            if (cnt == t.size()) {
                if (res.empty() || i - j + 1 < res.size())
                    res = s.substr(j, i - j + 1);
            }
        }

        return res;
    }
};
```

##### **Python**

```python
"""
先想清楚暴力写法如何写：
暴力枚举所有子串，查找有没有包含t里的所有字母。（双指针 一定要具备单调性 才可以用）
优化：对于某一个固定的r 都可以找到唯一的最近的l：当r往后走（值变大）r对应的l 一定不会向前走（你不会变小），l也是一定往后走的（值变大）==> 这就是具备单调性的。（简单证明：反证法）

另外一个问题：如何快速判断当前字符是否包含了t中的所有字符？
非常巧妙的地方：1. 用一个hash表统计t里每个字符出现的次数，用另外一个hash表统计窗口[l,r]内每个字符出现的次数;
2. 接着用一个变量cnt统计t里有多少个字符已经被包含了，当cnt==len(t)，说明窗口已经包含了t里所有字符；
3.考虑r往前走的时候，字符是否能被记录到cnt中；考虑什么时候l可以往前移动。

时间复杂度：每个字符只会遍历一次，而且哈希表操作常数次，所以是O(N)
"""

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        hs, ht = collections.defaultdict(int), collections.defaultdict(int)
        for c in t:
            ht[c] += 1
        cnt, l = 0, 0
        res = ''
        for r in range(len(s)):
            hs[s[r]] += 1
            if hs[s[r]] <= ht[s[r]]:
                cnt += 1
            while l <= r and hs[s[l]] > ht[s[l]]:  # 踩坑！l <= r 记得写上等于！
                hs[s[l]] -= 1
                l += 1
            if cnt == len(t):
                if not res or r - l + 1 < len(res):
                    res = s[l:r + 1]
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 209. 长度最小的子数组](https://leetcode-cn.com/problems/minimum-size-subarray-sum/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int minSubArrayLen(int s, vector<int>& nums) {
        int res = INT_MAX;
        for (int i = 0, j = 0, sum = 0; i < nums.size(); ++i) {
            sum += nums[i];
            while (sum - nums[j] >= s) sum -= nums[j++];
            if (sum >= s) res = min(res, i - j + 1);
        }
        if (res == INT_MAX) return 0;
        return res;
    }
    int minSubArrayLen_2(int s, vector<int>& nums) {
        int n = nums.size(), res = INT_MAX;
        int l = 0, r = 0, sum = 0;
        while (r < n) {
            sum += nums[r ++ ];
            while (sum >= s) {
                res = min(res, r - l);
                sum -= nums[l ++ ];
            }
        }
        return res == INT_MAX ? 0 : res;
    }
};
```

##### **Python**

```python
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        res = float('inf')
        l = 0; sumn = 0
        for r in range(len(nums)):
            sumn += nums[r]
            while l <= r and sumn - nums[l] >= s:
                sumn -= nums[l]
                l += 1
            if sumn >= s:
                res = min(res, r - l + 1)
        return res if res != float('inf') else 0
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 220. 存在重复元素 III](https://leetcode-cn.com/problems/contains-duplicate-iii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    typedef long long LL;

    bool containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t) {
        multiset<LL> S;
        S.insert(1e18), S.insert(-1e18);
        for (int i = 0, j = 0; i < nums.size(); i ++ ) {
            if (i - j > k) S.erase(S.find(nums[j ++ ]));
            int x = nums[i];
            auto it = S.lower_bound(x);
            if (*it - x <= t) return true;
            -- it;
            if (x - *it <= t) return true;
            S.insert(x);
        }
        return false;
    }

    
    bool containsNearbyAlmostDuplicate_2(vector<int>& nums, int k, int t) {
        multiset<LL> hash;
        multiset<LL>::iterator it;
        for (int i = 0; i < nums.size(); ++ i ) {
            it = hash.lower_bound((LL)nums[i] - t);
            if (it != hash.end() && *it <= (LL)nums[i] + t) return true;
            hash.insert(nums[i]);
            if (i >= k) hash.erase(hash.find(nums[i - k]));
        }
        return false;
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

> [!NOTE] **[LeetCode 239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        deque<int> q;
        vector<int> res;
        for (int i = 0; i < nums.size(); i ++ ) {
            if (q.size() && q.front() < i - k + 1) q.pop_front();
            while (q.size() && nums[i] >= nums[q.back()]) q.pop_back();
            q.push_back(i);
            if (i >= k - 1) res.push_back(nums[q.front()]);
        }
        return res;
    }
};
```

##### **Python**

```python
# 单调队列的经典应用
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        from collections import deque
        q = deque()
        res = []
        for r in range(len(nums)):
            if q and q[0] <= r - k:
                q.popleft()
            while q and nums[q[-1]] < nums[r]:
                q.pop()
            q.append(r)
            if r >= k - 1:
                res.append(nums[q[0]])
        return res  
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 395. 至少有K个重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-with-at-least-k-repeating-characters/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> trick + 滑动窗口

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int K;
    unordered_map<char, int> cnt;

    void add(char c, int & x, int & y) {
        if (!cnt[c]) x ++ ;
        cnt[c] ++ ;
        if (cnt[c] == K) y ++ ;
    }
    void del(char c, int & x, int & y) {
        if (cnt[c] == K) y -- ;
        cnt[c] -- ;
        if (!cnt[c]) x -- ;
    }
    int longestSubstring(string s, int k) {
        K = k;
        int res = 0;
        // 枚举应包含多少种字符
        for (int d = 1; d <= 26; ++ d ) {
            cnt.clear();
            // x: 字符种类数
            // y: 刚好出现 K 次的字符种类数
            for (int i = 0, j = 0, x = 0, y = 0; i < s.size(); ++ i ) {
                add(s[i], x, y);
                while (x > d) del(s[j ++ ], x, y);
                if (x == y) res = max(res, i - j + 1);
            }
        }
        return res;
    }
};
```

##### **Python**

```python
# r往后走的时候，l有可能往前走，这个不满足单调性，所以直接做 不单调
# 不单调怎么办，我们考虑枚举一些条件，使得变得单调
# ！！！精髓：枚举 区间中最多包含的不同字符的数量。 不同的字符 只有26个，所以枚举26次就可以了。
# 在扫描的时候，最多包含的字符的数量就确定了，比如k个。l一定是离r越远越好，那肯定某种字符的数量更多。对于每个终点r, 需要找到一个最左边的l使得[l,r]之前最多包含k个字符。（这个过程 就有单调性了）
# 维护：不同字符的数量，满足要求的字符数量。（用哈希表来维护）

import collections
class Solution:   
    def longestSubstring(self, s: str, k: int) -> int:
        max_len = 0
        for num in range(1, 27): # 枚举子串包含多少个不同的字符
            cnt = collections.defaultdict(int)
            l = 0
            for r in range(len(s)):
                cnt[s[r]] += 1
                while len(cnt) > num: # 移动左指针，找最长的合法区间
                    cnt[s[l]] -= 1
                    if cnt[s[l]] == 0:
                        del cnt[s[l]]
                    l += 1
                if len(cnt) == num:#此时区间有l个不同的字符，判断是否满足每个字符都至少出现k次
                    valid = True
                    for c in cnt:
                        if cnt[c] < k:
                            valid = False
                    if valid:
                        max_len = max(max_len, r - l + 1)
        return max_len
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 424. 替换后的最长重复字符](https://leetcode-cn.com/problems/longest-repeating-character-replacement/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int characterReplacement(string s, int k) {
        int res = 0;
        for (auto c = 'A'; c <= 'Z'; ++ c ) {
            for (int l = 0, r = 0, cnt = 0; r < s.size(); ++ r ) {
                if (s[r] == c) ++ cnt ;
                while (r - l + 1 - cnt > k) {
                    if (s[l] == c) -- cnt;
                    ++ l ;
                }
                res = max(res, r - l + 1);
            }
        }
        return res;
    }
};
```

##### **Python**

```python
# (题解区的 题解不严谨，没有解释为什么)
# 更严谨的做法：1. 先枚举出现最多的字符是什么（枚举26个大写字母）；2. 后续的双指针就很好想到。
# 对于每一个指针r, 找到最靠左的指针l，使得这个区间内 不是c字符的次数 <= k; 整个过程只需要维护c的次数

class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        res = 0 
        for c in range(ord('A'), ord('Z') + 1):
            c = chr(c)
            l, r = 0, 0
            cnt = 0
            while r < len(s):
                if s[r] == c:
                    cnt += 1
                while r - l + 1 - cnt > k:
                    if s[l] == c:
                        cnt -= 1
                    l += 1
                res = max(res, r - l + 1)
                r += 1
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 438. 找到字符串中所有字母异位词](https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        vector<int> res;
        unordered_map<char, int> need;
        for (auto c : p) ++ need[c];
        int size = need.size();
        for (int l = 0, r = 0, tot = 0; r < s.size(); ++ r ) {
            if ( -- need[s[r]] == 0) ++ tot;
            while (l <= r && need[s[l]] < 0) ++ need[s[l ++ ]];
            if (size == tot && r - l + 1 == p.size()) res.push_back(l);
        }
        return res;
    }

    vector<int> findAnagrams_1(string s, string p) {
        vector<int> res;
        unordered_map<char, int> need;
        for (auto c : p) ++ need[c];
        int size = need.size();
        for (int l = 0, r = 0, tot = 0; r < s.size(); ++ r ) {
            if ( -- need[s[r]] == 0) ++ tot;
            while (r - l + 1 > p.size()) {
                if (need[s[l]] == 0) -- tot;
                ++ need[s[l ++ ]];
            }
            if (tot == size) res.push_back(l);
        }
        return res;
    }

    vector<int> findAnagrams_2(string s, string p) {
        vector<int> res;
        unordered_map<char, int> need, cnt;
        for (auto c : p) ++ need[c];

        for (int l = 0, r = 0; r < s.size(); ++ r ) {
            ++ cnt[s[r]];
            while (l <= r && cnt[s[l]] > need[s[l]]) -- cnt[s[l ++ ]];
            if (cnt == need && r - l + 1 == p.size()) res.push_back(l);
        }
        return res;
    }
};
```

##### **Python**

```python
class Solution(object):
    def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
        """
        if not s or not p or len(s) < len(p):
            return []

        res = []
        sc = [0] * 26
        pc = [0] * 26

        # 先把pc，sc的hashmap给建好，并且先进行第一次比较
        for i in range(len(p)):
            sc[ord(s[i]) - ord("a")] += 1
            pc[ord(p[i]) - ord("a")] += 1

        if sc == pc:
            res.append(0)

        pLen = len(p)
        for i in range(pLen, len(s)):
            # 右指针+= 1
            sc[ord(s[i]) - ord("a")] += 1

            # 左指针-= 1
            sc[ord(s[i - pLen]) - ord("a")] -= 1

            # 假如两个哈希表对的上，则append左指针的位置
            if sc == pc:
                res.append(i - pLen + 1)
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 567. 字符串的排列](https://leetcode-cn.com/problems/permutation-in-string/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    unordered_map<char, int> need, has;

    bool check(char c) {
        return need.count(c) && has[c] == need[c];
    }

    bool checkInclusion(string s1, string s2) {
        for (auto c : s1) ++ need[c];
        for (int r = 0, l = 0, cnt = 0; r < s2.size(); ++ r ) {
            if (check(s2[r])) -- cnt;
            ++ has[s2[r]];
            if (check(s2[r])) ++ cnt;
            while (l <= r - int(s1.size())) {
                if (check(s2[l])) -- cnt;
                -- has[s2[l]];
                if (check(s2[l])) ++ cnt;
                ++ l;
            }
            if (cnt == need.size()) return true;
        }
        return false;
    }
};
```

##### **Python**

```python
# 用哈希表统计s1中字符出现的次数，用size表示字符种类；
# 在s2中 固定s1字符数量 即为 窗口大小，遍历s2中的字符，同时也用一个哈希表维护s2中字符出现的次数。最后判断 s2中字符次数 满足 s1字符次数的 个数 是否和 size相等 即可。

import collections
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        def check(c):
            if hs1[c] == hs2[c]: # 如果c在hs1中不存在，也会自动创建使得hs1[c] == 0，这样会让种类变多。
                return True
            return False
            
        hs1, hs2 = collections.defaultdict(int), collections.defaultdict(int)
        for c in s1:
            hs1[c] += 1
        size = len(hs1) # size表示有多少种字符，防止后续比较的时候hs1里的种类发生变化！
        l, cnt = 0, 0 # cnt表示 窗口内当前字符的个数 和 s1中的字符个数相等
        for r in range(len(s2)):
            if check(s2[r]):
                cnt -= 1
            hs2[s2[r]] += 1
            if check(s2[r]):
                cnt += 1
            
            if r - l >= len(s1):
                if check(s2[l]):
                    cnt -= 1
                hs2[s2[l]] -= 1
                if check(s2[l]):
                    cnt += 1
                l += 1 # 踩坑 不要忘了写了！
            if cnt == size:
                return True
        return False
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 643. 子数组最大平均数 I](https://leetcode-cn.com/problems/maximum-average-subarray-i/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    double findMaxAverage(vector<int>& nums, int k) {
        double res = -1e5;
        for (int i = 0, j = 0, s = 0; i < nums.size(); ++ i ) {
            s += nums[i];
            if (i - j + 1 > k) s -= nums[j ++ ];
            if (i >= k - 1) res = max(res, s / (double)k);
        }
        return res;
    }

    double findMaxAverage_2(vector<int>& nums, int k) {
        int n = nums.size();
        vector<int> s(n + 1);
        for (int i = 1; i <= n; ++ i ) s[i] = s[i - 1] + nums[i - 1];

        double res = INT_MIN;
        for (int i = k; i <= n; ++ i )
            res = max(res, (double)(s[i] - s[i - k]) / k);
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

> [!NOTE] **[LeetCode 1151. 最少交换次数来组合所有的 1](https://leetcode-cn.com/problems/minimum-swaps-to-group-all-1s-together/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 转化为定长区间内 求1个数最多是多少

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int minSwaps(vector<int>& data) {
        int tot = 0;
        for (auto v : data)
            if (v)
                ++ tot ;
        int res = -1e9, s = 0;
        for (int i = 0; i < data.size(); ++ i ) {
            s += data[i];
            if (i >= tot - 1) {
                res = max(res, s);
                s -= data[i - tot + 1];
            }
        }
        return tot - res;
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

> [!NOTE] **[LeetCode 1176. 健身计划评估](https://leetcode-cn.com/problems/diet-plan-performance/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 窗口即可 优雅写法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using LL = long long;
    int res, l, u;
    LL s;
    
    void f() {
        if (s < l) -- res;
        else if (s > u) ++ res;
    }
    
    int dietPlanPerformance(vector<int>& c, int k, int lower, int upper) {
        int n = c.size();
        res = 0, s = 0; l = lower, u = upper;
        for (int i = 0; i < k; ++ i ) s += c[i];
        f();
        
        for (int i = 1; i + k - 1 < n; ++ i ) {
            s -= c[i - 1];
            s += c[i + k - 1];
            f();
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

> [!NOTE] **[LeetCode 1208. 尽可能使字符串相等](https://leetcode-cn.com/problems/get-equal-substrings-within-budget/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然可以滑动窗口
> 
> **还有个构造前缀和二分的思维**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int equalSubstring(string s, string t, int maxCost) {
        int n = s.size();
        vector<int> cost(n);
        for (int i = 0; i < n; ++ i ) cost[i] = abs(s[i] - t[i]);
        int c = 0, res = 0;
        for (int l = 0, r = 0; r < n; ++ r ) {
            c += cost[r];
            while (l <= r && c > maxCost) {
                c -= cost[l];
                ++ l ;
            }
            res = max(res, r - l + 1);
        }
        return res;
    }
};
```

##### **C++ 二分**

```cpp
// liuzhou_101
class Solution {
public:
    int equalSubstring(string s, string t, int maxCost) {
        int n = s.size();
        vector<int> a(n + 1);
        int ret = 0;
        for (int i = 1; i <= n; ++ i ) {
            a[i] = a[i - 1] + abs(s[i - 1] - t[i - 1]);
            int j = lower_bound(a.begin(), a.begin() + i + 1, a[i] - maxCost) - a.begin();
            ret = max(ret, i - j);
        }
        return ret;
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

> [!NOTE] **[LeetCode 1425. 带限制的子序列和](https://leetcode-cn.com/problems/constrained-subsequence-sum/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 间隔小于等于 k 的子序列和最大值
> 
> 暴力 $O(n*k)$ 超时 可以滑动窗口维护每个 idx 前面 k 个数
> 
> 下面是借助 multiset 实现 找到前 k 个的最大值

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int n;
    int maxv = INT_MIN, v;
    int constrainedSubsetSum(vector<int>& nums, int k) {
        n = nums.size();
        vector<int> dp(n + 1);
        int maxv = INT_MIN;
        multiset<int> s;
        for (int i = 0; i < n; ++i) {
            dp[i] = nums[i];
            if (s.size() > k) s.erase(dp[i - k - 1]);
            if (!s.empty()) {
                auto it = --s.end();
                dp[i] = max(dp[i], nums[i] + *it);
            }
            maxv = max(maxv, dp[i]);
            s.insert(dp[i]);
        }

        return maxv;
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

> [!NOTE] **[LeetCode 1438. 绝对差不超过限制的最长连续子数组](https://leetcode-cn.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> multiset

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int longestSubarray(vector<int>& nums, int limit) {
        int n = nums.size(), ans = 0;
        multiset<int> ms;
        for (int L = 0, R = 0; L < n; ++L) {
            while (R < n) {
                ms.insert(nums[R]);
                if (*ms.rbegin() - *ms.begin() > limit) {
                    ms.erase(ms.find(nums[R]));
                    break;
                }
                ++R;
            }
            ans = max(R - L, ans);
            ms.erase(ms.find(nums[L]));
        }
        return ans;
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

> [!NOTE] **[LeetCode 1477. 找两个和为目标值且不重叠的子数组](https://leetcode-cn.com/problems/find-two-non-overlapping-sub-arrays-each-with-target-sum/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 自己比赛做法：记录前缀和 及该前缀和对应下标 存储合法区间到 ve 对ve排序后贪心即可 注意贪心条件：
> 
> `(p.second - 1 >= t1 || p.second - 1 + p.first <= s1)` 因为 第二个长度较长的区间 可以在 第一个长度较短区间 下标的前面。
> 
> **滑动窗口 + dp**

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int f[100005];  // 每个以i结尾的 最短满足target的长度
    int minSumOfLengths(vector<int>& arr, int target) {
        int i = 0, j = -1, cursum = 0, n = arr.size();
        int res = 100005;

        for (int j = 0; j < n; ++j) {
            cursum += arr[j];
            while (cursum > target) cursum -= arr[i++];  // 维护窗口
            int cur = 100005;
            if (cursum == target) {
                cur = j - i + 1;
                if (i) res = min(f[i - 1] + cur, res);
            }
            int pre = j && f[j - 1] < 100005 ? f[j - 1] : 100005;
            f[j] = min(cur, pre);
        }
        return res < 100005 ? res : -1;
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

> [!NOTE] **[LeetCode 1695. 删除子数组的最大得分](https://leetcode-cn.com/problems/maximum-erasure-value/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 简单滑动窗口
> 
> 小优化：前缀和数组可以省略，直接在窗口滑动时计算总和即可

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    int maximumUniqueSubarray(vector<int>& nums) {
        unordered_map<int, int> hash;
        int n = nums.size(), res = 0;
        vector<int> s(n + 1);
        for (int i = 1; i <= n; ++ i ) s[i] = s[i - 1] + nums[i - 1];
        for (int l = 0, r = 0, mul = 0; r < n; ++ r ) {
            ++ hash[nums[r]];
            if (hash[nums[r]] == 2) ++ mul;
            
            while (mul && l < r) {
                -- hash[nums[l]];
                if (hash[nums[l]] == 1) -- mul;
                ++ l;
            }
            res = max(res, s[r + 1] - s[l]);
        }
        return res;
    }
};
```

##### **C++ 小优化**

```cpp
class Solution {
public:
    int maximumUniqueSubarray(vector<int>& nums) {
        unordered_map<int, int> hash;
        int res = 0;
        for (int i = 0, j = 0, s = 0; i < nums.size(); i ++ ) {
            int x = nums[i];
            hash[x] ++ ;
            s += x;
            while (hash[x] > 1) {
                s -= nums[j];
                hash[nums[j ++ ]] -- ;
            }
            res = max(res, s);
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

> [!NOTE] **[LeetCode 2106. 摘水果](https://leetcode-cn.com/problems/maximum-fruits-harvested-after-at-most-k-steps/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 一开始想的是 trick ，考虑到最终合法范围一定是在 startPos 的左右侧：可能其中一侧重叠。
> 
> 随后根据重叠部分计算得到一个合法值，维护过程取 max 。
> 
> **但显然无需关注【在某个区间取某个值】，只需关注【取某个值时可能有哪些区间】即可** ==> 滑动窗口双指针
> 
> 要能想到滑动窗口性质 加强

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    // trick 双指针
    const static int N = 2e5 + 10;
    
    int w[N];
    
    int maxTotalFruits(vector<vector<int>>& fruits, int startPos, int k) {
        memset(w, 0, sizeof w);
        for (auto & f : fruits)
            w[f[0]] += f[1];
        
        int res = 0;
        for (int l = 0, r = 0, s = 0; l <= startPos && r < N; ++ r ) {
            s += w[r];
            
            // 直接推理取某个区间的 min 消耗，而非枚举区间计算消耗
            while (l <= r && r - l + min(abs(startPos - l), abs(r - startPos)) > k)
                s -= w[l], l ++ ;
            res = max(res, s);
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