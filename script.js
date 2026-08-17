/**
 * Satya Image Forensics — Frontend Client Logic
 */

(function () {
    "use strict";

    // API Configuration
    const API_ENDPOINT = "http://172.16.45.118:8000/api/verify";

    const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB

    // State Variables
    let selectedFile = null;
    let latestAnalysisResult = null;
    let loadingInterval = null;

    // DOM Elements
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const dropContentEmpty = document.getElementById("drop-content-empty");
    const dropContentPreview = document.getElementById("drop-content-preview");
    const previewImage = document.getElementById("preview-image");
    const previewFilename = document.getElementById("preview-filename");
    const previewFilesize = document.getElementById("preview-filesize");
    const removeImgBtn = document.getElementById("remove-img-btn");

    const screenshotToggle = document.getElementById("screenshot-toggle");
    const claimedSourceInput = document.getElementById("claimed-source-input");
    const analyzeBtn = document.getElementById("analyze-btn");

    const intakePanel = document.getElementById("intake-panel");
    const loadingPanel = document.getElementById("loading-panel");
    const resultsPanel = document.getElementById("results-panel");
    const loadingStepText = document.getElementById("loading-step-text");

    const errorBanner = document.getElementById("error-banner");
    const errorMessage = document.getElementById("error-message");
    const closeErrorBtn = document.getElementById("close-error-btn");

    const verdictBanner = document.getElementById("verdict-banner");
    const verdictTitle = document.getElementById("verdict-title");
    const verdictIcon = document.getElementById("verdict-icon");

    const confidenceCircle = document.getElementById("confidence-circle");
    const confidenceNumber = document.getElementById("confidence-number");
    const confidenceTier = document.getElementById("confidence-tier");
    const verdictReason = document.getElementById("verdict-reason");
    const provenanceBadges = document.getElementById("provenance-badges");

    // Sub-signal elements
    const elaBadge = document.getElementById("ela-badge");
    const elaBar = document.getElementById("ela-bar");
    const elaDesc = document.getElementById("ela-desc");

    const aiBadge = document.getElementById("ai-badge");
    const aiBar = document.getElementById("ai-bar");
    const aiDesc = document.getElementById("ai-desc");

    const dfBadge = document.getElementById("df-badge");
    const dfBar = document.getElementById("df-bar");
    const dfDesc = document.getElementById("df-desc");

    const chyronBadge = document.getElementById("chyron-badge");
    const chyronBar = document.getElementById("chyron-bar");
    const chyronDesc = document.getElementById("chyron-desc");

    const latencyVal = document.getElementById("latency-val");
    const copyJsonBtn = document.getElementById("copy-json-btn");
    const resetBtn = document.getElementById("reset-btn");

    // =========================================================================
    // Initialization & Event Listeners
    // =========================================================================

    function init() {
        // Drag & drop triggers
        dropZone.addEventListener("click", (e) => {
            if (e.target !== removeImgBtn) {
                fileInput.click();
            }
        });

        dropZone.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                fileInput.click();
            }
        });

        ["dragenter", "dragover"].forEach((eventName) => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.add("drag-active");
            });
        });

        ["dragleave", "drop"].forEach((eventName) => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.remove("drag-active");
            });
        });

        dropZone.addEventListener("drop", (e) => {
            const files = e.dataTransfer?.files;
            if (files && files.length > 0) {
                handleFileSelection(files[0]);
            }
        });

        fileInput.addEventListener("change", () => {
            if (fileInput.files && fileInput.files.length > 0) {
                handleFileSelection(fileInput.files[0]);
            }
        });

        removeImgBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            clearSelectedFile();
        });

        analyzeBtn.addEventListener("click", executeAnalysis);
        resetBtn.addEventListener("click", resetToUploadView);
        closeErrorBtn.addEventListener("click", hideError);
        copyJsonBtn.addEventListener("click", copyResultJson);
    }

    // =========================================================================
    // File Handling & Preview
    // =========================================================================

    function handleFileSelection(file) {
        hideError();

        // Validate MIME type
        const validTypes = ["image/jpeg", "image/png", "image/webp", "image/gif"];
        if (!validTypes.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png|webp|gif)$/i)) {
            showError("Please upload a valid image file (JPG, PNG, WEBP, or GIF).");
            return;
        }

        // Validate File Size
        if (file.size > MAX_FILE_SIZE_BYTES) {
            showError("File size exceeds 10MB limit. Please choose a smaller image.");
            return;
        }

        selectedFile = file;

        // Render preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewFilename.textContent = file.name;
            previewFilesize.textContent = formatBytes(file.size);

            dropContentEmpty.classList.add("hidden");
            dropContentPreview.classList.remove("hidden");
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function clearSelectedFile() {
        selectedFile = null;
        fileInput.value = "";
        previewImage.src = "";
        dropContentPreview.classList.add("hidden");
        dropContentEmpty.classList.remove("hidden");
        analyzeBtn.disabled = true;
    }

    function formatBytes(bytes) {
        if (bytes === 0) return "0 Bytes";
        const k = 1024;
        const sizes = ["Bytes", "KB", "MB", "GB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
    }

    // =========================================================================
    // Analysis API Request & Loading Animation
    // =========================================================================

    async function executeAnalysis() {
        if (!selectedFile) return;

        hideError();
        showLoadingState();

        const startTime = performance.now();
        const formData = new FormData();

        // Support both "image_file" and "file" field conventions
        formData.append("image", selectedFile);

        if (screenshotToggle.checked) {
            formData.append("is_screenshot", "true");
        }

        const claimedSource = claimedSourceInput.value.trim();
        if (claimedSource) {
            formData.append("claimed_source_url", claimedSource);
        }

        try {
            const response = await fetch(API_ENDPOINT, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                let errDetail = `Server returned HTTP ${response.status}`;
                try {
                    const errJson = await response.json();
                    if (errJson.detail) {
                        errDetail = errJson.detail;
                    }
                } catch (_) {}
                throw new Error(errDetail);
            }

            const data = await response.json();
            const measuredDurationMs = Math.round(performance.now() - startTime);

            latestAnalysisResult = data;
            renderResults(data, measuredDurationMs);

        } catch (error) {
            console.error("Forensics analysis failed:", error);
            stopLoadingAnimation();
            intakePanel.classList.remove("hidden");
            loadingPanel.classList.add("hidden");
            showError(error.message || "Failed to connect to image forensics API.");
        }
    }

    function showLoadingState() {
        intakePanel.classList.add("hidden");
        resultsPanel.classList.add("hidden");
        loadingPanel.classList.remove("hidden");

        const steps = [
            "Running Error Level Analysis (ELA)...",
            "Inspecting EXIF & generative provenance tags...",
            "Scanning 2D FFT spectral frequency artifacts...",
            "Analyzing facial boundary seams & symmetry...",
            "Evaluating chyron rendering & edge sharpness...",
            "Fusing detector signals into calibrated verdict..."
        ];

        let stepIdx = 0;
        loadingStepText.textContent = steps[0];

        loadingInterval = setInterval(() => {
            stepIdx = (stepIdx + 1) % steps.length;
            loadingStepText.textContent = steps[stepIdx];
        }, 500);
    }

    function stopLoadingAnimation() {
        if (loadingInterval) {
            clearInterval(loadingInterval);
            loadingInterval = null;
        }
    }

    // =========================================================================
    // Results Rendering & UI Polish
    // =========================================================================

    function renderResults(data, measuredDurationMs) {
        stopLoadingAnimation();

        loadingPanel.classList.add("hidden");
        resultsPanel.classList.remove("hidden");

        // 1. Verdict mapping
        const rawVerdict = (data.verdict || "uncertain").toLowerCase();
        let verdictKey = "uncertain";
        let verdictLabel = "UNCERTAIN";
        let verdictIconChar = "⚠️";
        let ringColor = "#F59E0B";

        if (rawVerdict.includes("authentic") || rawVerdict.includes("true")) {
            verdictKey = "authentic";
            verdictLabel = "AUTHENTIC";
            verdictIconChar = "✅";
            ringColor = "#22C55E";
        } else if (rawVerdict.includes("manipulated") || rawVerdict.includes("false")) {
            verdictKey = "manipulated";
            verdictLabel = "MANIPULATED";
            verdictIconChar = "❌";
            ringColor = "#EF4444";
        }

        verdictBanner.className = `verdict-banner verdict-${verdictKey}`;
        verdictTitle.textContent = verdictLabel;
        verdictIcon.textContent = verdictIconChar;

        // 2. Confidence percentage & progress circle
        let confPercent = 50;
        if (typeof data.confidence === "number") {
            confPercent = data.confidence <= 1.0
                ? Math.round(data.confidence * 100)
                : Math.round(data.confidence);
        }

        animateCounter(confidenceNumber, confPercent, 800);

        // SVG circle stroke-dashoffset (circumference = 2 * PI * 50 = 314.159)
        const circumference = 314.159;
        const offset = circumference - (confPercent / 100 * circumference);
        confidenceCircle.style.stroke = ringColor;
        confidenceCircle.style.strokeDashoffset = offset;

        // Confidence tier
        if (confPercent >= 80) {
            confidenceTier.textContent = "High Confidence";
        } else if (confPercent >= 60) {
            confidenceTier.textContent = "Moderate Confidence";
        } else {
            confidenceTier.textContent = "Low / Inconclusive";
        }

        // 3. Explanation text
        verdictReason.textContent = data.reason || data.explanation || data.explanation_en || "Analysis complete.";

        // 4. Provenance Badges
        renderProvenanceBadges(data);

        // 5. Sub-Signals Breakdown
        renderSubSignals(data);

        // 6. Processing Time
        const displayLatency = data.processing_time_ms || measuredDurationMs || 0;
        latencyVal.textContent = `${displayLatency}ms`;
    }

    function renderProvenanceBadges(data) {
        provenanceBadges.innerHTML = "";
        const signals = data.signals || {};
        const meta = signals.metadata || {};

        const tags = [];

        if (meta.editing_tools && meta.editing_tools.length > 0) {
            tags.push(`🛠️ ${meta.editing_tools.join(", ")}`);
        }
        if (meta.genai_tools && meta.genai_tools.length > 0) {
            tags.push(`✨ ${meta.genai_tools.join(", ")}`);
        }
        if (meta.is_stripped) {
            tags.push("📁 EXIF Stripped");
        } else if (meta.editing_tools?.length === 0 && meta.genai_tools?.length === 0) {
            tags.push("📷 Camera EXIF Intact");
        }

        if (signals.deepfake?.face_detected) {
            tags.push(`👤 ${signals.deepfake.face_count || 1} Face(s) Detected`);
        }

        if (signals.chyron_tampering?.is_screenshot) {
            tags.push(`📱 Screenshot (${signals.chyron_tampering.aspect_ratio || "Aspect"})`);
        }

        tags.forEach((text) => {
            const pill = document.createElement("span");
            pill.className = "provenance-pill";
            pill.textContent = text;
            provenanceBadges.appendChild(pill);
        });
    }

    function renderSubSignals(data) {
        const signals = data.signals || {};

        // ELA Signal
        const elaScore = signals.ela?.spatial_anomaly_score ?? signals.ela_score ?? 0.0;
        const elaPercent = Math.round(elaScore * 100);
        const anomBlocks = signals.ela?.anomalous_blocks ?? 0;
        updateSignalBar(
            elaBadge, elaBar, elaDesc,
            elaPercent,
            elaScore > 0.65 ? "badge-danger" : (elaScore > 0.4 ? "badge-warning" : "badge-clean"),
            elaScore > 0.65
                ? `High compression variance disparity (${anomBlocks} spliced block clusters)`
                : (elaScore > 0.4 ? "Moderate compression variance across block grid" : "Uniform JPEG compression history across entire image")
        );

        // AI Generator Signal
        const aiScore = signals.ai_generator?.ai_confidence ?? signals.ai_generated_score ?? 0.0;
        const aiPercent = Math.round(aiScore * 100);
        const modelUsed = signals.ai_generator?.model_used || "spectral_heuristics";
        updateSignalBar(
            aiBadge, aiBar, aiDesc,
            aiPercent,
            aiScore > 0.65 ? "badge-danger" : (aiScore > 0.4 ? "badge-warning" : "badge-clean"),
            aiScore > 0.65
                ? `Synthetic generation patterns detected (${modelUsed})`
                : `Natural optical capture signature (${modelUsed})`
        );

        // Face & Deepfake Signal
        const dfData = signals.deepfake || {};
        if (dfData.face_detected && !dfData.skipped) {
            const dfScore = dfData.deepfake_score ?? signals.deepfake_score ?? 0.0;
            const dfPercent = Math.round(dfScore * 100);
            updateSignalBar(
                dfBadge, dfBar, dfDesc,
                dfPercent,
                dfScore > 0.65 ? "badge-danger" : (dfScore > 0.4 ? "badge-warning" : "badge-clean"),
                dfScore > 0.65
                    ? `Facial blending seams & smoothing disparity detected (${dfData.face_count} face(s))`
                    : `Natural facial lighting & skin texture consistency (${dfData.face_count} face(s))`
            );
        } else {
            updateSignalBar(
                dfBadge, dfBar, dfDesc,
                0, "badge-neutral",
                "Skipped — no human face regions detected in image"
            );
        }

        // Chyron / Screenshot Tampering Signal
        const chyronData = signals.chyron_tampering || {};
        const chyronScore = chyronData.tamper_score ?? signals.chyron_score ?? 0.0;
        const chyronPercent = Math.round(chyronScore * 100);
        if (chyronData.is_screenshot) {
            updateSignalBar(
                chyronBadge, chyronBar, chyronDesc,
                chyronPercent,
                chyronScore > 0.45 ? "badge-danger" : "badge-clean",
                chyronScore > 0.45
                    ? `Mismatched edge sharpness in lower-third banner (doctored text)`
                    : `Text rendering & antialiasing consistent with screenshot frame`
            );
        } else {
            updateSignalBar(
                chyronBadge, chyronBar, chyronDesc,
                chyronPercent, "badge-neutral",
                "Non-screenshot aspect ratio — standard raster photo"
            );
        }
    }

    function updateSignalBar(badgeEl, barEl, descEl, percent, badgeClass, description) {
        badgeEl.textContent = `${percent}%`;
        badgeEl.className = `signal-badge ${badgeClass}`;

        barEl.style.width = `${Math.max(4, percent)}%`;

        if (badgeClass === "badge-danger") {
            barEl.style.background = "linear-gradient(90deg, #EF4444, #F87171)";
        } else if (badgeClass === "badge-warning") {
            barEl.style.background = "linear-gradient(90deg, #F59E0B, #FBBF24)";
        } else if (badgeClass === "badge-clean") {
            barEl.style.background = "linear-gradient(90deg, #22C55E, #4ADE80)";
        } else {
            barEl.style.background = "rgba(255, 255, 255, 0.2)";
        }

        descEl.textContent = description;
    }

    function animateCounter(element, targetVal, durationMs) {
        const startVal = 0;
        const startTime = performance.now();

        function update(now) {
            const progress = Math.min((now - startTime) / durationMs, 1);
            const current = Math.round(startVal + (targetVal - startVal) * easeOutQuad(progress));
            element.textContent = current;
            if (progress < 1) {
                requestAnimationFrame(update);
            }
        }

        requestAnimationFrame(update);
    }

    function easeOutQuad(t) {
        return t * (2 - t);
    }

    // =========================================================================
    // Utilities & Error Handling
    // =========================================================================

    function showError(msg) {
        errorMessage.textContent = msg;
        errorBanner.classList.remove("hidden");
    }

    function hideError() {
        errorBanner.classList.add("hidden");
    }

    function resetToUploadView() {
        resultsPanel.classList.add("hidden");
        loadingPanel.classList.add("hidden");
        intakePanel.classList.remove("hidden");
        clearSelectedFile();
    }

    function copyResultJson() {
        if (!latestAnalysisResult) return;
        const text = JSON.stringify(latestAnalysisResult, null, 2);

        navigator.clipboard.writeText(text).then(() => {
            const btnSpan = copyJsonBtn.querySelector("span");
            const originalText = btnSpan.textContent;
            btnSpan.textContent = "Copied! ✓";
            setTimeout(() => {
                btnSpan.textContent = originalText;
            }, 2000);
        }).catch((err) => {
            console.error("Clipboard copy failed:", err);
        });
    }

    // Run on DOM Ready
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
