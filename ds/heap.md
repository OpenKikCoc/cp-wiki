

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

## 习题

### heap

> [!NOTE] **[LeetCode 373. 查找和最小的K对数字](https://leetcode-cn.com/problems/find-k-pairs-with-smallest-sums/)**
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
    typedef tuple<int, int, int> tpi3;
    vector<vector<int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) {
        if (nums1.empty() || nums2.empty()) return {};
        int n = nums1.size(), m = nums2.size();
        priority_queue<tpi3, vector<tpi3>, greater<tpi3>> heap;
        for (int i = 0; i < m; ++ i )
            heap.push({nums2[i] + nums1[0], 0, i});
        vector<vector<int>> res;
        while (k -- && heap.size()) {
            auto [v, p1, p2] = heap.top(); heap.pop();
            res.push_back({nums1[p1], nums2[p2]});
            if (p1 + 1 < n)
                heap.push({nums1[p1 + 1] + nums2[p2], p1 + 1, p2});
        }
        return res;
    }
};

// tuple 也可以直接用 vector
// yxc
class Solution {
public:
    typedef vector<int> VI;
    vector<vector<int>> kSmallestPairs(vector<int>& a, vector<int>& b, int k) {
        if (a.empty() || b.empty()) return {};
        int n = a.size(), m = b.size();
        priority_queue<VI, vector<VI>, greater<VI>> heap;
        for (int i = 0; i < m; i ++ ) heap.push({b[i] + a[0], 0, i});
        vector<VI> res;
        while (k -- && heap.size()) {
            auto t = heap.top();
            heap.pop();
            res.push_back({a[t[1]], b[t[2]]});
            if (t[1] + 1 < n)
                heap.push({a[t[1] + 1] + b[t[2]], t[1] + 1, t[2]});
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

> [!NOTE] **[LeetCode 658. 找到 K 个最接近的元素](https://leetcode-cn.com/problems/find-k-closest-elements/)**
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
    vector<int> findClosestElements(vector<int>& arr, int k, int x) {
        priority_queue<pair<int, int>> heap;
        for (auto v: arr) {
            heap.push({abs(x - v), v});
            if (heap.size() > k) heap.pop();
        }
        vector<int> res;
        while (heap.size()) res.push_back(heap.top().second), heap.pop();
        sort(res.begin(), res.end());
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

> [!NOTE] **[LeetCode 692. 前K个高频单词](https://leetcode-cn.com/problems/top-k-frequent-words/)**
> 
> 题意: 尝试以 $O(n log k)$ 时间复杂度和 $O(n)$ 空间复杂度解决。

> [!TIP] **思路**
> 
> 原地建堆 复杂度要求

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using PIS = pair<int, string>;
    
    vector<string> topKFrequent(vector<string>& words, int k) {
        unordered_map<string, int> cnt;
        for (auto& w: words) cnt[w] ++ ;

        vector<PIS> ws;
        for (auto [k, v]: cnt) ws.push_back({v, k});
        auto cmp = [](PIS a, PIS b) {
            if (a.first != b.first) return a.first < b.first;
            return a.second > b.second;
        };
        make_heap(ws.begin(), ws.end(), cmp);
        
        vector<string> res;
        while (k -- ) {
            res.push_back(ws[0].second);
            pop_heap(ws.begin(), ws.end(), cmp);
            ws.pop_back();
        }

        return res;
    }
};
```

##### **Python**

```python
# 1. 用哈希表统计每个单词出现的次数； 2. 原地建堆，取前k个；
class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        # my_cnt = collections.Counter(words) 也可以直接用计数器 进行计数。 
        my_dict = collections.defaultdict(int)
        for c in words:
            my_dict[c] += 1 
        
        q = []
        for key, val in my_dict.items():
            heapq.heappush(q, (-val, key))  #  堆的元素可以是元组/列表 类型; 小根堆 所以 取负数
        res = []
        for _ in range(k):
            res.append(heapq.heappop(q)[1])
        return res
```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1439. 有序矩阵中的第 k 个最小数组和](https://leetcode-cn.com/problems/find-the-kth-smallest-sum-of-a-matrix-with-sorted-rows/)** [TAG]
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 任意一行选一个数 求第k小
> 
> 解决办法：
> 
> - 二分数组和 求个数 个数比k个大就缩小右边界 ==> 并不友好
> 
> - 维护小顶堆 每次取出一个组合就把这个后面可拓展的组合全部加入堆 使用 set 对组合去重 ==> **正解**（参考算法竞赛进阶指南做法

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
struct dwell {
    vector<int> pos;
    int sum;
    dwell(const vector<int>& _pos, int _sum) : pos(_pos), sum(_sum) {}
    bool operator<(const dwell& that) const { return sum > that.sum; }
};

class Solution {
public:
    int kthSmallest(vector<vector<int>>& mat, int k) {
        int m = mat.size(), n = mat[0].size();

        auto getsum = [&](const vector<int>& v) {
            int ret = 0;
            for (int i = 0; i < m; ++i) { ret += mat[i][v[i]]; }
            return ret;
        };

        set<vector<int>> seen;

        vector<int> init(m, 0);
        seen.insert(init);
        priority_queue<dwell> q;
        q.emplace(init, getsum(init));
        for (int _ = 0; _ < k - 1; ++_) {
            auto [pos, sum] = q.top();
            q.pop();
            for (int i = 0; i < m; ++i) {
                if (pos[i] + 1 < n) {
                    ++pos[i];
                    if (!seen.count(pos)) {
                        q.emplace(pos, getsum(pos));
                        seen.insert(pos);
                    }
                    --pos[i];
                }
            }
        }

        auto fin = q.top();
        return fin.sum;
    }
};
// zerotrac2
```

##### **Python**

```python

```

<!-- tabs:end -->
</details>

<br>

* * *

> [!NOTE] **[LeetCode 1801. 积压订单中的订单总数](https://leetcode-cn.com/problems/number-of-orders-in-the-backlog/)**
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
    const int MOD = 1e9 + 7;
    using PII = pair<int, int>;
    using LL = long long;
    int getNumberOfBacklogOrders(vector<vector<int>>& orders) {
        priority_queue<PII, vector<PII>, greater<PII>> sells;
        priority_queue<PII, vector<PII>, less<PII>> buys;
        int n = orders.size();
        for (int i = 0; i < n; ++ i ) {
            auto & os = orders[i];
            int cnt = os[1], type = os[2], price = os[0];
            if (type) {
                while (buys.size() && buys.top().first >= price) {
                    auto [t_price, t_cnt] = buys.top(); buys.pop();
                    int cost = min(cnt, t_cnt);
                    cnt -= cost, t_cnt -= cost;
                    if (t_cnt) {
                        buys.push({t_price, t_cnt});
                        break;
                    }
                }
                // 1 sell
                if (cnt)
                    sells.push({price, cnt});
            } else {
                while (sells.size() && sells.top().first <= price) {
                    auto [t_price, t_cnt] = sells.top(); sells.pop();
                    int cost = min(cnt, t_cnt);
                    cnt -= cost, t_cnt -= cost;
                    if (t_cnt) {
                        sells.push({t_price, t_cnt});
                        break;
                    }
                }
                // 0 buy
                if (cnt)
                    buys.push({price, cnt});
            }
        }
        
        LL res = 0;
        while (buys.size()) {
            auto [p, c] = buys.top(); buys.pop();
            res = (res + c) % MOD;
        }
        while (sells.size()) {
            auto [p, c] = sells.top(); sells.pop();
            res = (res + c) % MOD;
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

> [!NOTE] **[LeetCode 1847. 最近的房间](https://leetcode-cn.com/problems/closest-room/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然堆维护即可
> 
> 最初写的时候 `lower_bound(S.begin(), S.end(), pid)` TLE
> 
> 注意 STL set 使用 `S.lower_bound(pid)`

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++**

```cpp
class Solution {
public:
    using PII = pair<int, int>;
    using TIII = tuple<int, int, int>;
    
    vector<int> closestRoom(vector<vector<int>>& rooms, vector<vector<int>>& queries) {
        vector<PII> ve;
        for (auto & r : rooms)
            ve.push_back({r[1], r[0]});
        sort(ve.begin(), ve.end());
        reverse(ve.begin(), ve.end());
        
        vector<TIII> qs;
        int m = queries.size();
        for (int i = 0; i < m; ++ i )
            qs.push_back({queries[i][1], queries[i][0], i});
        sort(qs.begin(), qs.end());
        // 反向 面积从大到小 条件越来越严格
        reverse(qs.begin(), qs.end());
        
        vector<int> res(m, -1);
        int p = 0, n = ve.size();
        set<int> S;
        for (auto & [msz, pid, id] : qs) {
            while (p < n && ve[p].first >= msz) {
                S.insert(ve[p].second);
                p ++ ;
            }
            if (S.size()) {
                auto it = S.lower_bound(pid); // ATTENTION lower_bound(S.begin(), S.end(), pid); 就会超时
                int t = 2e9;
                if (it != S.end()) {
                    if (abs(*it - pid) < abs(t - pid))
                        t = *it;
                }
                if (it != S.begin()) {
                    it -- ;
                    if (abs(*it - pid) <= abs(t - pid))
                        t = *it;
                }
                res[id] = t;
            }
        }
        return res;
    }
};
```

##### **C++ trick**

```cpp
class Solution {
public:
    // using PII = pair<int, int>;
    // using TIII = tuple<int, int, int>;
    
    vector<int> closestRoom(vector<vector<int>>& rooms, vector<vector<int>>& queries) {
        auto cmp = [&](auto & a, auto & b){
            return a[1] > b[1];
        };
        sort(rooms.begin(), rooms.end(), cmp);
        int m = queries.size();
        for (int i = 0; i < m; ++ i )
            queries[i].push_back(i);
        sort(queries.begin(), queries.end(), cmp);
        
        vector<int> res(m, -1);
        int p = 0, n = rooms.size();
        set<int> S;
        for (auto & q : queries) {
            int pid = q[0], msz = q[1], id = q[2];
            while (p < n && rooms[p][1] >= msz) {
                S.insert(rooms[p][0]);
                p ++ ;
            }
            if (S.size()) {
                auto it = S.lower_bound(pid);
                int t = 2e9;
                if (it != S.end()) {
                    if (*it - pid < t - pid)
                        t = *it;
                }
                if (it != S.begin()) {
                    it -- ;
                    if (pid - *it <= t - pid)
                        t = *it;
                }
                res[id] = t;
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

### 双堆模拟 一般用于占用类问题

> [!NOTE] **[LeetCode 1882. 使用服务器处理任务](https://leetcode-cn.com/problems/process-tasks-using-servers/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 显然堆
> 
> 一开始WA: 问题在于同一时刻开始多个任务的实现细节

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ WA**

```cpp
// WA 30 / 34
class Solution {
public:
    using PII = pair<int, int>;
    
    vector<int> assignTasks(vector<int>& servers, vector<int>& tasks) {
        int n = servers.size(), m = tasks.size();
        vector<int> res(m, -1);
        
        priority_queue<PII, vector<PII>, greater<PII>> idle, working;
        priority_queue<int, vector<int>, greater<int>> task;
        
        for (int i = 0; i < n; ++ i )
            idle.push({servers[i], i});
        
        for (int i = 0; i < m; ++ i ) {
            // 当前遍历到第i个任务 时间也为i 弹出所有到i时刻已完成的任务服务器
            while (working.size()) {
                auto [et, sid] = working.top();
                if (et > i)
                    break;
                else {
                    working.pop();
                    idle.push({servers[sid], sid});
                }
            }
            // 加入当前第i个任务
            task.push(i);
            // 无可用服务器 continue
            if (idle.empty())
                continue;
            
            // 取下标最小的任务
            auto tid = task.top(); task.pop();
            int t = tasks[tid];

            // 存在可用服务器 即使用最优的服务器
            auto [rank, sid] = idle.top(); idle.pop();
            res[tid] = sid;
            working.push({i + t, sid});
        }
        
        // now时间戳
        int now = m;
        while (task.size()) {
            // 根据时间戳更新服务器状态
            while (working.size()) {
                auto [et, sid] = working.top();
                if (et > now)
                    break;
                else {
                    working.pop();
                    idle.push({servers[sid], sid});
                }
            }
            
            // 取下标最小的任务
            auto tid = task.top();
            int t = tasks[tid];
            
            // 如果有可用服务器 则使用该服务器
            // 否则更新时间戳 快进到会有服务器被弹出的时刻
            if (idle.size()) {
                task.pop();
                auto [rank, sid] = idle.top(); idle.pop();
                res[tid] = sid;
                working.push({now + t, sid});
            } else {
                auto [et, sid] = working.top();
                now = et;
            }
        }
        return res;
    }
};
```

##### **C++ AC**

```cpp
class Solution {
public:
    using PII = pair<int, int>;
    
    vector<int> assignTasks(vector<int>& servers, vector<int>& tasks) {
        int n = servers.size(), m = tasks.size();
        vector<int> res(m, -1);
        
        priority_queue<PII, vector<PII>, greater<PII>> idle, working;
        priority_queue<int, vector<int>, greater<int>> task;
        
        for (int i = 0; i < n; ++ i )
            idle.push({servers[i], i});
        
        for (int i = 0; i < m; ++ i ) {
            // 当前遍历到第i个任务 时间也为i 弹出所有到i时刻已完成的任务服务器
            while (working.size()) {
                auto [et, sid] = working.top();
                if (et > i)
                    break;
                else {
                    working.pop();
                    idle.push({servers[sid], sid});
                }
            }
            // 加入当前第i个任务
            task.push(i);
            // 无可用服务器 continue
            if (idle.empty())
                continue;
            
            // ===== 问题在于 此时此刻可能有多个任务可以同时开始 =====
            // 取下标最小的任务
            while (task.size() && idle.size()) {
                auto tid = task.top(); task.pop();
                int t = tasks[tid];

                // 存在可用服务器 即使用最优的服务器
                auto [rank, sid] = idle.top(); idle.pop();
                res[tid] = sid;
                working.push({i + t, sid});
            }
        }
        
        // now时间戳
        int now = m;
        while (task.size()) {
            // 根据时间戳更新服务器状态
            while (working.size()) {
                auto [et, sid] = working.top();
                if (et > now)
                    break;
                else {
                    working.pop();
                    idle.push({servers[sid], sid});
                }
            }
            
            // 取下标最小的任务
            auto tid = task.top();
            int t = tasks[tid];
            
            // 如果有可用服务器 则使用该服务器
            // 否则更新时间戳 快进到会有服务器被弹出的时刻
            if (idle.size()) {
                task.pop();
                auto [rank, sid] = idle.top(); idle.pop();
                res[tid] = sid;
                working.push({now + t, sid});
            } else {
                auto [et, sid] = working.top();
                now = et;
            }
        }
        return res;
    }
};
```

##### **C++ 标准**

```cpp
class Solution {
public:
    using PII = pair<int, int>;
    
    vector<int> assignTasks(vector<int>& servers, vector<int>& tasks) {
        int n = servers.size(), m = tasks.size();
        
        priority_queue<PII, vector<PII>, greater<PII>> busy, idle;
        
        for (int i = 0; i < n; ++ i )
            idle.push({servers[i], i});
        
        int ts = 0;
        vector<int> res(m);
        for (int i = 0; i < m; ++ i ) {
            ts = max(ts, i);
            while (busy.size() && busy.top().first <= ts) {
                auto [_, sid] = busy.top(); busy.pop();
                idle.push({servers[sid], sid});
            }
            
            if (idle.empty()) {
                ts = busy.top().first;
                i = i - 1;  // 更新时间戳 再次检查本个任务
            } else {
                auto [rank, sid] = idle.top(); idle.pop();
                res[i] = sid;
                busy.push({ts + tasks[i], sid});
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

> [!NOTE] **[LeetCode 2402. 会议室 III](https://leetcode.cn/problems/meeting-rooms-iii/)**
> 
> 题意: TODO

> [!TIP] **思路**
> 
> 标准做法是双堆模拟

<details>
<summary>详细代码</summary>
<!-- tabs:start -->

##### **C++ 暴力**

```cpp
class Solution {
public:
    using LL = long long;
    const static int N = 110;
    
    LL last[N], cnt[N]; // start end 比较大 要用 LL
    
    void init() {
        memset(last, 0, sizeof last);
        memset(cnt, 0, sizeof cnt);
    }
    
    int mostBooked(int n, vector<vector<int>>& meetings) {
        init();
        sort(meetings.begin(), meetings.end());
        
        for (auto & m : meetings) {
            int l = m[0], r = m[1];
            int p = -1;
            for (int i = 0; i < n; ++ i )
                if (last[i] <= l) {
                    p = i;
                    break;
                }
            if (p == -1) {
                for (int i = 0; i < n; ++ i )
                    if (p == -1 || last[i] < last[p])
                        p = i;
                last[p] += r - l;
            } else {
                last[p] = r;
            }
            cnt[p] ++ ;
        }
        
        int p = -1;
        for (int i = 0; i < n; ++ i )
            if (p == -1 || cnt[i] > cnt[p])
                p = i;
        return p;
    }
};
```

##### **C++ 双堆模拟**

```cpp
class Solution {
public:
    using LL = long long;
    using PLI = pair<LL, int>;
    const static int N = 110;

    int cnt[N];

    int mostBooked(int n, vector<vector<int>>& meetings) {
        sort(meetings.begin(), meetings.end());
        priority_queue<int, vector<int>, greater<>> idle;
        priority_queue<PLI, vector<PLI>, greater<>> working;
        
        for (int i = 0; i < n; ++ i )
            idle.push(i);
        
        for (auto & m : meetings) {
            LL l = m[0], r = m[1], p = -1;
            while (!working.empty() && working.top().first <= l)
                idle.push(working.top().second), working.pop();
            if (idle.empty()) {
                auto [e, i] = working.top(); working.pop();
                r = e + r - l;
                p = i;
            } else {
                p = idle.top(); idle.pop();
            }
            cnt[p] ++ ;
            working.push({r, p});
        }
        int p = -1;
        for (int i = 0; i < n; ++ i )
            if (p == -1 || cnt[i] > cnt[p])
                p = i;
        return p;
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

### multiset

> [!NOTE] **[LeetCode 218. 天际线问题](https://leetcode-cn.com/problems/the-skyline-problem/)**
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
    vector<vector<int>> getSkyline(vector<vector<int>>& buildings) {
        vector<pair<int, int>> h;
        multiset<int> m;
        vector<vector<int>> res;
        for (auto &b : buildings) {
            h.push_back({b[0], -b[2]});
            h.push_back({b[1], b[2]});
        }
        sort(h.begin(), h.end());
        m.insert(0);
        int prev = 0;
        for (auto i : h) {
            if (i.second < 0)
                m.insert(-i.second);
            else
                m.erase(m.find(i.second));
            int cur = *m.rbegin();
            if (cur != prev) {
                res.push_back({i.first, cur});
                prev = cur;
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

> [!NOTE] **[LeetCode 480. 滑动窗口中位数](https://leetcode-cn.com/problems/sliding-window-median/)**
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
    int k;
    // r.size() >= l.size()
    multiset<int> left, right;

    double get_medium() {
        if (k % 2) return *right.begin();
        // 这里写double 处理整数溢出
        return ((double)*left.rbegin() + *right.begin()) / 2;
    }
    vector<double> medianSlidingWindow(vector<int>& nums, int _k) {
        k = _k;
        for (int i = 0; i < k; ++ i ) right.insert(nums[i]);
        // 较小的一半压入left
        for (int i = 0; i < k / 2; ++ i ) {
            left.insert(*right.begin());
            right.erase(right.begin());
        }
        vector<double> res;
        res.push_back(get_medium());

        for (int i = k; i < nums.size(); ++ i ) {
            int x = nums[i], y = nums[i - k];

            if (x >= *right.begin()) right.insert(x);
            else left.insert(x);
            if (y >= *right.begin()) right.erase(right.find(y));
            else left.erase(left.find(y));

            while (left.size() > right.size()) {
                right.insert(*left.rbegin());
                // rbegin 必须写find
                left.erase(left.find(*left.rbegin()));
            }
            while (left.size() + 1 < right.size()) {
                left.insert(*right.begin());
                right.erase(right.begin());
            }
            res.push_back(get_medium());
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