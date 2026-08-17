document.getElementById('verify-btn').addEventListener('click', async () => {
  const text = document.getElementById('text-input').value.trim();
  const imageInput = document.getElementById('image-input');
  const imageFile = imageInput ? imageInput.files[0] : null;
  const langSelect = document.getElementById('lang-select');
  const language = langSelect ? langSelect.value : 'tamil';

  if (!text && !imageFile) {
    alert('Please enter a text claim or upload an image.');
    return;
  }

  showLoadingState();

  const formData = new FormData();
  if (text) formData.append('text', text);
  if (imageFile) formData.append('image', imageFile);
  formData.append('language', language);

  try {
    const res = await fetch('/api/verify', { method: 'POST', body: formData });
    if (!res.ok) throw new Error('Server error');
    const data = await res.json();
    renderVerdictCard(data);
  } catch (err) {
    showError('Something went wrong. Please try again.');
  }
});

document.getElementById('reset-btn')?.addEventListener('click', () => {
  document.getElementById('text-input').value = '';
  const imageInput = document.getElementById('image-input');
  if (imageInput) imageInput.value = '';
  showIntakeView();
});

function renderVerdictCard(data) {
  const card = document.getElementById('verdict-card');
  card.classList.remove('verdict-true', 'verdict-false', 'verdict-unverifiable');
  card.classList.add(`verdict-${data.verdict}`);

  document.getElementById('verdict-title').textContent = {
    'true': 'Likely True ✅',
    'false': 'Likely False ❌',
    'unverifiable': 'Unverifiable ⚠️'
  }[data.verdict] || 'Unknown';

  const confEl = document.getElementById('confidence');
  if (confEl) {
    confEl.textContent = data.verdict === 'unverifiable' || !data.confidence
      ? 'Confidence: N/A'
      : `Confidence: ${data.confidence}%`;
  }

  const langSelect = document.getElementById('lang-select');
  const selectedLang = langSelect ? langSelect.value : 'english';

  const enEl = document.getElementById('explanation-en');
  const regionalEl = document.getElementById('explanation-regional');

  if (selectedLang === 'english') {
    if (enEl) enEl.textContent = data.explanation_en || '';
    if (regionalEl) {
      regionalEl.textContent = '';
      regionalEl.style.display = 'none';
    }
  } else {
    if (enEl) enEl.textContent = '';
    if (regionalEl) {
      regionalEl.textContent = data.explanation_regional || '';
      regionalEl.style.display = 'block';
    }
  }

  const sourcesList = document.getElementById('sources');
  if (sourcesList) {
    sourcesList.innerHTML = (data.sources || []).map(s =>
      `<li><a href="${s.url}" target="_blank" rel="noopener">${s.title}</a></li>`
    ).join('') || '<li>No sources available</li>';
  }

  showVerdictView();
}

function showLoadingState() {
  document.getElementById('intake-view')?.classList.add('hidden');
  document.getElementById('verdict-view')?.classList.add('hidden');
  document.getElementById('loading-view')?.classList.remove('hidden');
}

function showVerdictView() {
  document.getElementById('loading-view')?.classList.add('hidden');
  document.getElementById('intake-view')?.classList.add('hidden');
  document.getElementById('verdict-view')?.classList.remove('hidden');
}

function showIntakeView() {
  document.getElementById('loading-view')?.classList.add('hidden');
  document.getElementById('verdict-view')?.classList.add('hidden');
  document.getElementById('intake-view')?.classList.remove('hidden');
}

function showError(msg) {
  document.getElementById('loading-view')?.classList.add('hidden');
  alert(msg);
  showIntakeView();
}
