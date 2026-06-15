// Global Application State
let activeMode = 'free';
let guidedStep = 1;
let guidedResponses = [];
let currentTopic = "";

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
    // Switch to free mode by default
    switchMode('free');
});

// Switch active view mode
function switchMode(mode) {
    activeMode = mode;
    
    // Toggle tab active class
    document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
    document.getElementById(`tab-${mode}`).classList.add("active");
    
    // Toggle panels visibility
    document.querySelectorAll(".mode-panel").forEach(panel => panel.classList.remove("active"));
    document.getElementById(`panel-${mode}`).classList.add("active");
    
    // In guided mode, load a topic if not already loaded
    if (mode === 'guided' && guidedResponses.length === 0 && !currentTopic) {
        loadGuidedTopic();
    }
}

// Reset state variables
function resetApp() {
    // Reset inputs
    document.getElementById("text-free").value = "";
    document.getElementById("text-guided").value = "";
    document.getElementById("words-free").innerText = "0";
    document.getElementById("words-guided").innerText = "0";
    
    // Reset guided state
    guidedStep = 1;
    guidedResponses = [];
    currentTopic = "";
    
    // Reset step tracker UI
    for (let i = 1; i <= 3; i++) {
        const dot = document.getElementById(`step-${i}`);
        dot.className = "step-dot";
        if (i === 1) dot.classList.add("active");
        
        if (i < 3) {
            document.getElementById(`line-${i}`).className = "step-line";
        }
    }
    
    document.getElementById("prompt-badge").style.display = "none";
    
    // Hide results, show panel
    document.getElementById("results").style.display = "none";
    document.querySelector(".main-panel").style.display = "block";
    
    // Re-load topic if guided
    if (activeMode === 'guided') {
        loadGuidedTopic();
    }
}

// Clear text fields
function clearText(areaId, countId) {
    document.getElementById(areaId).value = "";
    document.getElementById(countId).innerText = "0";
}

// Update word counts in real time
function updateWordCount(areaId, countId) {
    const text = document.getElementById(areaId).value.trim();
    const wordCount = text === "" ? 0 : text.split(/\s+/).length;
    document.getElementById(countId).innerText = wordCount;
}

// Fetch a new topic from the API
async function loadGuidedTopic() {
    try {
        const res = await fetch("/api/get-topic");
        const data = await res.json();
        currentTopic = data.topic;
        document.getElementById("topic-prompt").innerText = currentTopic;
    } catch (err) {
        console.error("Failed to load topic:", err);
        document.getElementById("topic-prompt").innerText = "Describe a moment in your life where you had to make a very tough logical choice. How did you feel?";
    }
}

// Show validation errors
function showError(msg) {
    document.getElementById("error-message").innerText = msg;
    document.getElementById("error-banner").style.display = "flex";
    document.getElementById("error-banner").scrollIntoView({ behavior: 'smooth' });
}

// Hide error banner
function hideError() {
    document.getElementById("error-banner").style.display = "none";
}

// Variant 1: Submit Free Text
async function submitFree(e) {
    e.preventDefault();
    hideError();
    
    const text = document.getElementById("text-free").value.trim();
    const wordCount = text === "" ? 0 : text.split(/\s+/).length;
    
    if (wordCount < 10) {
        showError("Please write a meaningful text of at least 10 words for analysis.");
        return;
    }
    
    showLoading(true);
    
    try {
        const res = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text })
        });
        
        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.error || "Prediction request failed.");
        }
        
        const data = await res.json();
        renderResults(data);
    } catch (err) {
        showError(err.message || "An error occurred while analyzing your text.");
    } finally {
        showLoading(false);
    }
}

// Variant 2: Submit Guided Text (Looping based on confidence)
async function submitGuided(e) {
    e.preventDefault();
    hideError();
    
    const text = document.getElementById("text-guided").value.trim();
    const wordCount = text === "" ? 0 : text.split(/\s+/).length;
    
    if (wordCount < 10) {
        showError("Please write a descriptive answer of at least 10 words.");
        return;
    }
    
    // Show button loading state
    const submitBtn = document.getElementById("btn-submit-guided");
    submitBtn.classList.add("loading");
    
    // Add current text to stack
    guidedResponses.push(text);
    
    // Concatenate all responses written so far for prediction
    const fullText = guidedResponses.join(" ");
    
    try {
        const res = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: fullText })
        });
        
        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.error || "Inference server error.");
        }
        
        const data = await res.json();
        
        // Evaluate confidence threshold (e.g. 22%)
        // If confidence is low AND we have not reached step 3, prompt for next topic
        const confidenceThreshold = 22.0;
        
        if (data.confidence < confidenceThreshold && guidedStep < 3) {
            // Increment guided step
            guidedStep++;
            
            // UI updates for the steps
            document.getElementById(`step-${guidedStep - 1}`).className = "step-dot completed";
            document.getElementById(`line-${guidedStep - 1}`).className = "step-line active";
            document.getElementById(`step-${guidedStep}`).className = "step-dot active";
            
            // Show low confidence warning badge
            document.getElementById("prompt-badge").style.display = "inline-flex";
            
            // Clear current textarea input
            clearText("text-guided", "words-guided");
            
            // Fetch next topic
            await loadGuidedTopic();
            
            // Scroll to topic
            document.querySelector(".topic-card").scrollIntoView({ behavior: 'smooth' });
        } else {
            // Confidence is high enough OR we reached step 3 -> Show final results!
            // Mark last step as completed
            document.getElementById(`step-${guidedStep}`).className = "step-dot completed";
            
            // Reveal dashboard
            renderResults(data);
        }
        
    } catch (err) {
        showError(err.message || "An error occurred during guided evaluation.");
        // Revert last response push on error
        guidedResponses.pop();
    } finally {
        submitBtn.classList.remove("loading");
    }
}

// Show/Hide page-level loading overlay
function showLoading(show) {
    const overlay = document.getElementById("loading-overlay");
    if (show) {
        overlay.classList.add("active");
    } else {
        overlay.classList.remove("active");
    }
}

// Render inference data on the dashboard UI
function renderResults(data) {
    // Hide inputs panel
    document.querySelector(".main-panel").style.display = "none";
    
    // Display Primary Type
    const primaryInfo = data.results_info[0];
    document.getElementById("res-mbti-code").innerText = data.primary_type;
    document.getElementById("res-mbti-title").innerText = primaryInfo.title;
    document.getElementById("res-mbti-desc").innerText = primaryInfo.description;
    document.getElementById("res-mbti-summary").innerText = primaryInfo.summary;
    
    // Confidence Meter
    document.getElementById("res-confidence-val").innerText = `${data.confidence}%`;
    document.getElementById("res-confidence-fill").style.width = `${data.confidence}%`;
    
    // Render Strengths and Blind Spots
    const strengthsUl = document.getElementById("res-strengths");
    strengthsUl.innerHTML = "";
    primaryInfo.strengths.forEach(s => {
        const li = document.createElement("li");
        li.innerText = s;
        strengthsUl.appendChild(li);
    });
    
    const weaknessesUl = document.getElementById("res-weaknesses");
    weaknessesUl.innerHTML = "";
    primaryInfo.weaknesses.forEach(w => {
        const li = document.createElement("li");
        li.innerText = w;
        weaknessesUl.appendChild(li);
    });

    // Multiple Probable Personalities (Alternative Types)
    const altContainer = document.getElementById("multi-types-container");
    const altList = document.getElementById("res-alt-list");
    altList.innerHTML = "";
    
    if (data.probable_types.length > 1) {
        altContainer.style.display = "block";
        // Iterate over alternative types (skip index 0 which is the primary)
        for (let i = 1; i < data.results_info.length; i++) {
            const alt = data.results_info[i];
            const pill = document.createElement("div");
            pill.className = "alt-type-pill";
            pill.innerHTML = `
                <span class="alt-code">${alt.mbti}</span>
                <span class="alt-title">${alt.title}</span>
            `;
            altList.appendChild(pill);
        }
    } else {
        altContainer.style.display = "none";
    }

    // Trait Dimension Fills
    // I/E Axis
    const ie = data.trait_percentages['I/E'];
    document.getElementById("fill-ie-left").style.width = `${ie.pos_percent}%`;
    document.getElementById("fill-ie-right").style.width = `${ie.neg_percent}%`;
    document.getElementById("label-ie-left").innerText = `${ie.pos_percent}%`;
    document.getElementById("label-ie-right").innerText = `${ie.neg_percent}%`;

    // N/S Axis
    const ns = data.trait_percentages['N/S'];
    document.getElementById("fill-ns-left").style.width = `${ns.pos_percent}%`;
    document.getElementById("fill-ns-right").style.width = `${ns.neg_percent}%`;
    document.getElementById("label-ns-left").innerText = `${ns.pos_percent}%`;
    document.getElementById("label-ns-right").innerText = `${ns.neg_percent}%`;

    // F/T Axis
    const ft = data.trait_percentages['F/T'];
    document.getElementById("fill-ft-left").style.width = `${ft.pos_percent}%`;
    document.getElementById("fill-ft-right").style.width = `${ft.neg_percent}%`;
    document.getElementById("label-ft-left").innerText = `${ft.pos_percent}%`;
    document.getElementById("label-ft-right").innerText = `${ft.neg_percent}%`;

    // J/P Axis
    const jp = data.trait_percentages['J/P'];
    document.getElementById("fill-jp-left").style.width = `${jp.pos_percent}%`;
    document.getElementById("fill-jp-right").style.width = `${jp.neg_percent}%`;
    document.getElementById("label-jp-left").innerText = `${jp.pos_percent}%`;
    document.getElementById("label-jp-right").innerText = `${jp.neg_percent}%`;

    // Linguistic explainability word markers
    renderWordMarkers('ie', data.influence_details['I/E'], 'i', 'e');
    renderWordMarkers('ns', data.influence_details['N/S'], 'n', 's');
    renderWordMarkers('ft', data.influence_details['F/T'], 'f', 't');
    renderWordMarkers('jp', data.influence_details['J/P'], 'j', 'p');

    // Reveal Results Section
    document.getElementById("results").style.display = "block";
    document.getElementById("results").scrollIntoView({ behavior: 'smooth' });
}

// Render explainability word tags for a dimension
function renderWordMarkers(axisId, detail, posLetter, negLetter) {
    const posContainer = document.getElementById(`markers-${axisId}-pos`);
    const negContainer = document.getElementById(`markers-${axisId}-neg`);
    
    posContainer.innerHTML = "";
    negContainer.innerHTML = "";
    
    // Add positive tags (Associated with I, N, F, J)
    if (detail.pos_markers.length > 0) {
        detail.pos_markers.forEach(item => {
            const tag = document.createElement("span");
            tag.className = `influence-tag tag-${posLetter}`;
            tag.innerText = item.word;
            tag.title = `Influence weight: +${item.weight}`;
            posContainer.appendChild(tag);
        });
    } else {
        posContainer.innerHTML = "<span style='font-size:0.85rem;color:var(--text-muted);font-style:italic;'>No prominent markers found.</span>";
    }
    
    // Add negative tags (Associated with E, S, T, P)
    if (detail.neg_markers.length > 0) {
        detail.neg_markers.forEach(item => {
            const tag = document.createElement("span");
            tag.className = `influence-tag tag-${negLetter}`;
            tag.innerText = item.word;
            tag.title = `Influence weight: ${item.weight}`;
            negContainer.appendChild(tag);
        });
    } else {
        negContainer.innerHTML = "<span style='font-size:0.85rem;color:var(--text-muted);font-style:italic;'>No prominent markers found.</span>";
    }
}
