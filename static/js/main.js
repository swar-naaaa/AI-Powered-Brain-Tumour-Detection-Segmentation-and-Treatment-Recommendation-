// Main JavaScript for Brain Tumor Detection System

// Global variables
let modelLoaded = false;
let currentAnalysis = null;

// Utility functions
function showMessage(message, type = 'info') {
    const container = document.getElementById('messageContainer');
    if (!container) return;
    
    const alertClass = type === 'success' ? 'alert alert-success' : 
                      type === 'error' ? 'alert alert-error' : 
                      'alert alert-info';
    
    const iconClass = type === 'success' ? 'bi-check-circle-fill' : 
                     type === 'error' ? 'bi-exclamation-circle-fill' : 
                     'bi-info-circle-fill';

    container.innerHTML = `
        <div class="${alertClass} fade-in">
            <p class="font-semibold flex items-center gap-2">
                <i class="bi ${iconClass}"></i>
                ${message}
            </p>
        </div>
    `;

    // Auto-hide after 5 seconds
    setTimeout(() => {
        container.innerHTML = '';
    }, 5000);
}

function showLoading(element, text = 'Loading...') {
    element.disabled = true;
    element.innerHTML = `<div class="loading-spinner mr-2"></div>${text}`;
}

function hideLoading(element, originalText) {
    element.disabled = false;
    element.innerHTML = originalText;
}

// Model loading functionality
async function loadModel() {
    const btn = document.getElementById('loadModelBtn');
    const status = document.getElementById('modelStatus');
    
    if (!btn || !status) return;
    
    showLoading(btn, 'Loading AI Model...');
    
    try {
        const response = await fetch('/load_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            modelLoaded = true;
            status.innerHTML = '<span class="text-green-600 font-semibold"><i class="bi bi-check-circle-fill mr-1"></i>Model loaded successfully!</span>';
            btn.style.display = 'none';
            
            const uploadSection = document.getElementById('uploadSection');
            if (uploadSection) {
                uploadSection.style.display = 'block';
                uploadSection.classList.add('fade-in');
            }
            
            showMessage('AI Model loaded successfully! You can now upload brain scans for analysis.', 'success');
        } else {
            status.innerHTML = '<span class="text-red-600 font-semibold"><i class="bi bi-x-circle-fill mr-1"></i>' + data.message + '</span>';
            hideLoading(btn, '<i class="bi bi-download mr-2"></i>Load AI Model');
            showMessage('Failed to load model: ' + data.message, 'error');
        }
    } catch (error) {
        console.error('Model loading error:', error);
        status.innerHTML = '<span class="text-red-600 font-semibold"><i class="bi bi-x-circle-fill mr-1"></i>Network error occurred</span>';
        hideLoading(btn, '<i class="bi bi-download mr-2"></i>Load AI Model');
        showMessage('Network error: ' + error.message, 'error');
    }
}

// Image handling functions
function previewImage(input) {
    if (input.files && input.files[0]) {
        const file = input.files[0];
        
        // Validate file type
        if (!file.type.startsWith('image/')) {
            showMessage('Please select a valid image file.', 'error');
            input.value = '';
            return;
        }
        
        // Validate file size (10MB limit)
        if (file.size > 10 * 1024 * 1024) {
            showMessage('File size must be less than 10MB.', 'error');
            input.value = '';
            return;
        }
        
        const reader = new FileReader();
        reader.onload = function(e) {
            const previewImg = document.getElementById('previewImg');
            const uploadPrompt = document.getElementById('uploadPrompt');
            const imagePreview = document.getElementById('imagePreview');
            
            if (previewImg && uploadPrompt && imagePreview) {
                previewImg.src = e.target.result;
                uploadPrompt.style.display = 'none';
                imagePreview.classList.remove('hidden');
                imagePreview.classList.add('fade-in');
            }
        };
        reader.readAsDataURL(file);
    }
}

function clearImage() {
    const imageInput = document.getElementById('imageInput');
    const uploadPrompt = document.getElementById('uploadPrompt');
    const imagePreview = document.getElementById('imagePreview');
    const resultsSection = document.getElementById('resultsSection');
    
    if (imageInput) imageInput.value = '';
    if (uploadPrompt) uploadPrompt.style.display = 'block';
    if (imagePreview) imagePreview.classList.add('hidden');
    if (resultsSection) resultsSection.classList.add('hidden');
    
    currentAnalysis = null;
}

// Analysis functionality
async function analyzeImage(formData) {
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Analysis error:', error);
        throw error;
    }
}

function displayResults(data) {
    currentAnalysis = data;
    
    // Show segmentation image
    const segmentationDiv = document.getElementById('segmentationImage');
    if (segmentationDiv && data.plot_image) {
        segmentationDiv.innerHTML = `
            <img src="data:image/png;base64,${data.plot_image}" 
                 class="w-full rounded-lg shadow-lg" 
                 alt="Brain Tumor Segmentation">
        `;
    }

    // Display treatment plan
    const treatmentDiv = document.getElementById('treatmentPlan');
    if (treatmentDiv && data.treatment_plan) {
        // Format the treatment plan with better styling
        const formattedPlan = data.treatment_plan
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n/g, '<br>');
        
        treatmentDiv.innerHTML = `
            <div class="treatment-plan">
                ${formattedPlan}
            </div>
        `;
    }

    // Show results section
    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
        resultsSection.classList.remove('hidden');
        resultsSection.classList.add('fade-in');
        
        // Scroll to results with smooth animation
        setTimeout(() => {
            resultsSection.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'start' 
            });
        }, 100);
    }
}

function startNewAnalysis() {
    clearImage();
    
    const uploadSection = document.getElementById('uploadSection');
    if (uploadSection) {
        setTimeout(() => {
            uploadSection.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'start' 
            });
        }, 100);
    }
}

// Drag and drop functionality
function initializeDragAndDrop() {
    const uploadArea = document.querySelector('.border-dashed');
    if (!uploadArea) return;
    
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        this.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        this.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        this.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const imageInput = document.getElementById('imageInput');
            if (imageInput) {
                imageInput.files = files;
                previewImage(imageInput);
            }
        }
    });
}

// Form submission handling
function initializeFormHandling() {
    const uploadForm = document.getElementById('uploadForm');
    if (!uploadForm) return;
    
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        if (!modelLoaded) {
            showMessage('Please load the AI model first.', 'error');
            return;
        }

        const fileInput = document.getElementById('imageInput');
        if (!fileInput || !fileInput.files[0]) {
            showMessage('Please select an image first.', 'error');
            return;
        }

        const analyzeBtn = document.getElementById('analyzeBtn');
        if (!analyzeBtn) return;
        
        showLoading(analyzeBtn, 'Analyzing...');

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        try {
            const data = await analyzeImage(formData);

            if (data.success) {
                displayResults(data);
                showMessage('Analysis completed successfully!', 'success');
            } else {
                showMessage('Analysis failed: ' + data.message, 'error');
            }
        } catch (error) {
            console.error('Analysis error:', error);
            showMessage('Network error during analysis: ' + error.message, 'error');
        } finally {
            hideLoading(analyzeBtn, '<i class="bi bi-search mr-2"></i>Analyze Brain Scan');
        }
    });
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeDragAndDrop();
    initializeFormHandling();
    
    // Add click handler for file input
    const selectImageBtn = document.querySelector('button[onclick*="imageInput"]');
    if (selectImageBtn) {
        selectImageBtn.addEventListener('click', function() {
            const imageInput = document.getElementById('imageInput');
            if (imageInput) imageInput.click();
        });
    }
    
    console.log('Brain Tumor Detection System initialized');
});

// Export functions for global access
window.loadModel = loadModel;
window.previewImage = previewImage;
window.clearImage = clearImage;
window.startNewAnalysis = startNewAnalysis;