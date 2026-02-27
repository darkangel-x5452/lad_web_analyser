# Quick Usage Guide - Enhanced Extractor

## 🚀 Fast Start (3 Steps)

### 1. Install
```bash
pip install -r requirements.txt --break-system-packages
python -m playwright install chromium
```

### 2. Test
```bash
python enhanced_demo.py
```

### 3. Use It!
```bash
# Fast (text-only)
python enhanced_extractor.py -u "URL" -q "your query"

# Accurate (with AI vision)
python enhanced_extractor.py -u "URL" -q "your query" --vision
```

---

## 📋 Common Use Cases

### Extract Sports Stats
```bash
python enhanced_extractor.py \
  -u "https://www.basketball-reference.com/teams/LAL/2024.html" \
  -q "Lakers points per game field goal percentage" \
  --vision \
  -o lakers_stats.md
```

### Extract Stock Data
```bash
python enhanced_extractor.py \
  -u "https://finance.yahoo.com/quote/AAPL" \
  -q "Apple stock price performance revenue" \
  --vision \
  -o apple_stock.md
```

### Extract Product Specs
```bash
python enhanced_extractor.py \
  -u "https://www.apple.com/iphone-15/specs/" \
  -q "iPhone 15 battery camera display specifications" \
  --vision \
  -o iphone_specs.md
```

### Extract Research Data
```bash
python enhanced_extractor.py \
  -u "https://arxiv.org/abs/1234.5678" \
  -q "model architecture results accuracy" \
  --vision \
  -o paper_results.md
```

### Quick Text Extraction (No Vision)
```bash
python enhanced_extractor.py \
  -u "https://blog.example.com/article" \
  -q "main topic key points" \
  -o article.md
```

---

## 🎯 Options Cheat Sheet

| Option | What It Does | Example |
|--------|--------------|---------|
| `-u URL` | Set URL to extract from | `-u "https://example.com"` |
| `-q "query"` | Set search query | `-q "statistics data"` |
| `-o file.md` | Save to file | `-o output.md` |
| `--vision` | Use AI vision analysis | `--vision` |
| `--screenshot path` | Save screenshot | `--screenshot page.png` |
| `--keep-screenshot` | Don't delete screenshot | `--keep-screenshot` |
| `--text-only` | Skip vision (faster) | `--text-only` |

---

## 🤔 When to Use What?

### Use `--vision` (AI Vision) When:
✅ Page has charts, graphs, or data tables  
✅ You need maximum accuracy  
✅ Visual layout matters  
✅ Statistics in styled boxes/cards  
✅ Complex data relationships  

### Skip Vision (Text-Only) When:
✅ Simple blog post or article  
✅ Speed is critical  
✅ Processing many URLs  
✅ No visual data on page  
✅ Text-heavy documentation  

---

## 💡 Pro Tips

1. **Start Simple**
   ```bash
   # Try text-only first to preview
   python enhanced_extractor.py -u URL -q "query"
   
   # Then add --vision if you need more detail
   python enhanced_extractor.py -u URL -q "query" --vision
   ```

2. **Be Specific in Queries**
   ```bash
   # ❌ Too vague
   -q "information"
   
   # ✅ Specific
   -q "Lakers 2024 points per game 3-point percentage"
   ```

3. **Save Screenshots for Later**
   ```bash
   python enhanced_extractor.py \
     -u URL -q "query" \
     --vision \
     --screenshot screenshot.png \
     --keep-screenshot
   ```

4. **Batch Processing**
   ```bash
   # Process multiple URLs (text-only for speed)
   for url in url1 url2 url3; do
     python enhanced_extractor.py -u $url -q "query" -o ${url##*/}.md
   done
   ```

---

## 🆘 Quick Troubleshooting

**Problem:** "Failed to capture screenshot"  
**Fix:** `python -m playwright install chromium --with-deps`

**Problem:** Vision analysis not working  
**Fix:** Make sure you're using `--vision` flag

**Problem:** Empty output  
**Fix:** Query too specific, try broader keywords

**Problem:** Too much irrelevant content  
**Fix:** Add more specific keywords to query

**Problem:** Installation errors  
**Fix:** Use `--break-system-packages` flag with pip

---

## 📊 Quick Comparison

| Mode | Speed | Accuracy | Cost | Best For |
|------|-------|----------|------|----------|
| Text-Only | ⚡⚡⚡ Fast | 75% | Free | Articles, docs |
| Vision | 🐢 Slower | 95% | ~$0.02 | Data, charts, tables |

---

## 🎓 Learning Path

1. **Day 1:** Run `python enhanced_demo.py` - try all examples
2. **Day 2:** Use text-only mode on 5 different websites
3. **Day 3:** Try vision mode on data-heavy pages
4. **Day 4:** Compare text vs vision results
5. **Day 5:** Integrate into your workflow

---

## 📚 Full Documentation

- **ENHANCED_README.md** - Complete guide with all features
- **COMPARISON_GUIDE.md** - Detailed text vs vision comparison
- **README.md** - Original text-only version documentation

---

## 🤝 Need Help?

1. Run the demo: `python enhanced_demo.py`
2. Read the comparison guide: `COMPARISON_GUIDE.md`
3. Check examples in the demo scripts

---

**Remember:** 
- Use text-only for speed 💨
- Use vision for accuracy 🎯
- Start simple, then enhance! ✨
