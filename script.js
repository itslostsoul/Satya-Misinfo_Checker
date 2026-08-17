// ==========================================
// SATYA - FORWARD CHECKER
// Frontend Logic
// ==========================================


// ==========================================
// 1. CONFIGURATION
// ==========================================

// TEMPORARY:
// Replace this with the actual backend URL
// once the backend team gives it to us.

const API_URL = "http://localhost:5000/api/check";


// Maximum time we allow the backend to respond
const REQUEST_TIMEOUT = 60000;


// ==========================================
// 2. GET HTML ELEMENTS
// ==========================================

const claimText = document.getElementById("claim-text");
const claimImage = document.getElementById("claim-image");
const verifyBtn = document.getElementById("verify-btn");

const intakeView = document.getElementById("intake-view");
const loadingView = document.getElementById("loading-view");
const resultView = document.getElementById("result-view");

const verdictContainer = document.getElementById("verdict-container");
const verdictTitle = document.getElementById("verdict-title");
const confidenceScore = document.getElementById("confidence-score");

const explanationEnglish = document.getElementById("expl-en");
const sourceLinks = document.getElementById("source-links");

const resetBtn = document.getElementById("reset-btn");


// ==========================================
// 3. STATE
// ==========================================

let selectedImage = null;
let countdownTimer = null;


// ==========================================
// 4. IMAGE SELECTION
// ==========================================

claimImage.addEventListener("change", function () {

    if (claimImage.files.length === 0) {
        selectedImage = null;
        return;
    }

    selectedImage = claimImage.files[0];

    console.log("Image selected:", selectedImage.name);

    // Change upload text to show selected file
    const uploadLabel = document.querySelector(".upload-label");

    uploadLabel.textContent = `📷 ${selectedImage.name}`;
});


// ==========================================
// 5. VERIFY BUTTON
// ==========================================

verifyBtn.addEventListener("click", async function () {

    const text = claimText.value.trim();

    // Check whether user provided anything
    if (!text && !selectedImage) {

        alert("Please paste a claim or upload an image.");

        return;
    }


    // Prevent multiple requests
    verifyBtn.disabled = true;


    // Show loading screen
    showLoading();


    try {

        const result = await sendToBackend(text, selectedImage);

        console.log("Backend response:", result);

        displayVerdict(result);

    } catch (error) {

        console.error("Verification error:", error);

        showError(error.message);

    } finally {

        verifyBtn.disabled = false;

    }

});


// ==========================================
// 6. SEND DATA TO BACKEND
// ==========================================

async function sendToBackend(text, image) {

    const formData = new FormData();


    // Add text if available
    if (text) {
        formData.append("text", text);
    }


    // Add image if available
    if (image) {
        formData.append("image", image);
    }


    // AbortController lets us stop the request
    // if it takes longer than 60 seconds.

    const controller = new AbortController();

    const timeout = setTimeout(() => {

        controller.abort();

    }, REQUEST_TIMEOUT);


    try {

        const response = await fetch(API_URL, {

            method: "POST",

            body: formData,

            signal: controller.signal

        });


        clearTimeout(timeout);


        // Backend returned an HTTP error
        if (!response.ok) {

            throw new Error(
                `Server error (${response.status})`
            );

        }


        // Convert JSON response into JavaScript object
        const data = await response.json();

        return data;

    } catch (error) {

        clearTimeout(timeout);


        if (error.name === "AbortError") {

            throw new Error(
                "The verification took too long. Please try again."
            );

        }


        throw error;

    }

}


// ==========================================
// 7. SHOW LOADING SCREEN
// ==========================================

function showLoading() {

    intakeView.classList.add("hidden");

    resultView.classList.add("hidden");

    loadingView.classList.remove("hidden");


    startCountdown();

}


// ==========================================
// 8. 60 SECOND COUNTDOWN
// ==========================================

function startCountdown() {

    let seconds = 60;

    const loadingText =
        document.querySelector(".loading-text");


    loadingText.textContent =
        `Scanning fact-check databases... ${seconds}s`;


    clearInterval(countdownTimer);


    countdownTimer = setInterval(() => {

        seconds--;


        if (seconds <= 0) {

            clearInterval(countdownTimer);

            loadingText.textContent =
                "Finalizing verification...";

            return;

        }


        loadingText.textContent =
            `Scanning fact-check databases... ${seconds}s`;

    }, 1000);

}


// ==========================================
// 9. DISPLAY VERDICT
// ==========================================

function displayVerdict(data) {

    clearInterval(countdownTimer);


    loadingView.classList.add("hidden");

    resultView.classList.remove("hidden");


    // --------------------------------------
    // Get verdict
    // --------------------------------------

    const verdict =
        normalizeVerdict(data.verdict);


    // --------------------------------------
    // Change card styling
    // --------------------------------------

    verdictContainer.classList.remove(
        "verdict-true",
        "verdict-false",
        "verdict-unverifiable"
    );


    verdictContainer.classList.add(
        `verdict-${verdict}`
    );


    // --------------------------------------
    // Verdict title
    // --------------------------------------

    if (verdict === "true") {

        verdictTitle.textContent =
            "Likely True ✅";

    }

    else if (verdict === "false") {

        verdictTitle.textContent =
            "Likely False ❌";

    }

    else {

        verdictTitle.textContent =
            "Unverifiable ⚠️";

    }


    // --------------------------------------
    // Confidence
    // --------------------------------------

    let confidence =
        Number(data.confidence);


    // Convert decimal confidence
    // e.g. 0.85 → 85

    if (confidence <= 1) {

        confidence = confidence * 100;

    }


    confidence = Math.round(confidence);


    confidenceScore.textContent =
        `${confidence}%`;


    // --------------------------------------
    // Explanation
    // --------------------------------------

    explanationEnglish.textContent =
        data.explanation ||
        data.explanation_en ||
        "No explanation was provided.";


    // --------------------------------------
    // Sources
    // --------------------------------------

    displaySources(data.sources);

}


// ==========================================
// 10. NORMALIZE VERDICT
// ==========================================

function normalizeVerdict(verdict) {

    if (!verdict) {

        return "unverifiable";

    }


    verdict = verdict
        .toString()
        .toLowerCase()
        .trim();


    if (
        verdict.includes("true") ||
        verdict.includes("likely true")
    ) {

        return "true";

    }


    if (
        verdict.includes("false") ||
        verdict.includes("likely false")
    ) {

        return "false";

    }


    return "unverifiable";

}


// ==========================================
// 11. DISPLAY SOURCES
// ==========================================

function displaySources(sources) {

    sourceLinks.innerHTML = "";


    // No sources
    if (!sources || sources.length === 0) {

        sourceLinks.textContent =
            "No reliable source found.";

        return;

    }


    sources.forEach((source, index) => {

        const link = document.createElement("a");

        // Support both formats:
        //
        // "https://example.com"
        //
        // OR
        //
        // { title: "...", url: "..." }

        if (typeof source === "string") {

            link.href = source;

            link.textContent =
                `Source ${index + 1}`;

        } else {

            link.href = source.url;

            link.textContent =
                source.title ||
                `Source ${index + 1}`;

        }


        link.target = "_blank";

        link.rel = "noopener noreferrer";


        if (index > 0) {

            sourceLinks.appendChild(
                document.createTextNode(" • ")
            );

        }


        sourceLinks.appendChild(link);

    });

}


// ==========================================
// 12. ERROR HANDLING
// ==========================================

function showError(message) {

    clearInterval(countdownTimer);


    loadingView.classList.add("hidden");

    resultView.classList.remove("hidden");


    verdictContainer.classList.remove(
        "verdict-true",
        "verdict-false"
    );


    verdictContainer.classList.add(
        "verdict-unverifiable"
    );


    verdictTitle.textContent =
        "Unable to Verify ⚠️";


    confidenceScore.textContent =
        "—";


    explanationEnglish.textContent =
        message ||
        "Something went wrong while checking this forward.";


    sourceLinks.textContent =
        "Please try again.";

}


// ==========================================
// 13. RESET BUTTON
// ==========================================

resetBtn.addEventListener("click", function () {

    clearInterval(countdownTimer);


    // Clear inputs
    claimText.value = "";

    claimImage.value = "";

    selectedImage = null;


    // Restore upload label
    const uploadLabel =
        document.querySelector(".upload-label");

    uploadLabel.textContent =
        "📷 Tap to Upload Image";


    // Restore views
    resultView.classList.add("hidden");

    loadingView.classList.add("hidden");

    intakeView.classList.remove("hidden");


    // Restore default card state
    verdictContainer.classList.remove(
        "verdict-true",
        "verdict-false"
    );

    verdictContainer.classList.add(
        "verdict-unverifiable"
    );

});
