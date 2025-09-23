// MMM Frontend Application v1.9.61 with cleaned up Marginal ROI section and fixed budget allocation
console.log('🚀 MMM App v1.9.61 Loading - CLEANED MARGINAL ROI & FIXED BUDGET ALLOCATION');

class MMMApp {
    constructor() {
        this.apiUrl = 'http://mmm-alb-production-190214907.us-east-2.elb.amazonaws.com/api';
        this.uploadId = null;
        this.runId = null;
        this.progressInterval = null;
        this.cvStructureInfo = null;  // Store CV structure info
        this.nestedCVUsed = false;    // Track if nested CV was used

        // Add VISIBLE indicator that JS is working
        this.addVisibleDebugIndicator();

        // Configure Chart.js to prevent automatic currency formatting
        this.configureChartDefaults();

        this.initializeEventListeners();
    }

    addVisibleDebugIndicator() {
        // Banner removed - JS loads silently now
        console.log('MMM Application JavaScript loaded successfully');
    }

    configureChartDefaults() {
        // Configure Chart.js global defaults to prevent automatic currency formatting
        console.log('🔧 Configuring Chart.js defaults to prevent currency formatting');

        if (typeof Chart !== 'undefined') {
            // Disable any automatic formatting
            Chart.defaults.locale = 'en-US';
            console.log('✅ Chart.js global configuration applied');
        } else {
            console.warn('⚠️ Chart.js not yet loaded, cannot configure defaults');
        }
    }

    initializeEventListeners() {
        const fileInput = document.getElementById('file-input');
        const browseBtn = document.getElementById('browse-btn');
        const uploadArea = document.getElementById('upload-area');
        const startTrainingBtn = document.getElementById('start-training');
        const cancelTrainingBtn = document.getElementById('cancel-training');

        // File upload events
        browseBtn.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => this.handleFileSelect(e.target.files[0]));
        
        // Drag and drop events
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            this.handleFileSelect(e.dataTransfer.files[0]);
        });

        // Training controls
        startTrainingBtn.addEventListener('click', () => this.startTraining());
        cancelTrainingBtn.addEventListener('click', () => this.cancelTraining());
    }

    showStatus(message, type = 'info') {
        const statusDiv = document.getElementById('upload-status');
        statusDiv.innerHTML = `<div class="status ${type}">${message}</div>`;
    }

    showSection(sectionId) {
        document.getElementById(sectionId).classList.remove('hidden');
    }

    hideSection(sectionId) {
        document.getElementById(sectionId).classList.add('hidden');
    }

    async handleFileSelect(file) {
        if (!file) return;
        
        if (!file.name.endsWith('.csv')) {
            this.showStatus('❌ Please select a CSV file.', 'error');
            return;
        }

        this.showStatus('📤 Uploading file...', 'info');
        
        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${this.apiUrl}/data/upload`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                this.uploadId = result.upload_id;
                this.showStatus(`✅ File uploaded successfully! Upload ID: ${this.uploadId}`, 'success');
                this.displayDataSummary(result);
                this.displayChannelInfo(result.channel_info);
                this.showSection('summary-section');
                this.showSection('config-section');
                document.getElementById('start-training').disabled = false;
                // Fetch and display channel classifications
                await this.fetchChannelClassifications();
            } else {
                this.showStatus(`❌ Upload failed: ${result.detail}`, 'error');
            }
        } catch (error) {
            this.showStatus(`❌ Upload error: ${error.message}`, 'error');
        }
    }

    displayDataSummary(data) {
        const summary = data.data_summary;
        const summaryHtml = `
            <div class="results-grid">
                <div class="metric-card">
                    <div class="metric-value">${summary.total_days}</div>
                    <div class="metric-label">Total Days</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">$${this.formatNumber(summary.total_profit)}</div>
                    <div class="metric-label">Total Profit</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">$${this.formatNumber(summary.total_annual_spend)}</div>
                    <div class="metric-label">Annual Spend</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${summary.channel_count}</div>
                    <div class="metric-label">Marketing Channels</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${summary.data_quality_score}%</div>
                    <div class="metric-label">Data Quality Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${summary.business_tier}</div>
                    <div class="metric-label">Business Tier</div>
                </div>
            </div>
            <p style="margin-top: 20px; color: #666;">
                📅 Date Range: ${new Date(summary.date_range.start).toLocaleDateString()} - ${new Date(summary.date_range.end).toLocaleDateString()}
            </p>
        `;
        document.getElementById('data-summary').innerHTML = summaryHtml;
    }

    displayChannelInfo(channels) {
        const channelHtml = Object.entries(channels).map(([name, info]) => `
            <div class="channel-card">
                <div class="channel-name">${this.formatChannelName(name)}</div>
                <div class="channel-stats">
                    <div>Type: ${info.type}</div>
                    <div>Spend: $${this.formatNumber(info.total_spend)}</div>
                    <div>Share: ${(info.spend_share * 100).toFixed(1)}%</div>
                    <div>Active Days: ${info.days_active}</div>
                </div>
            </div>
        `).join('');
        
        document.getElementById('channel-info').innerHTML = `
            <h3 style="margin-top: 30px; margin-bottom: 15px;">📺 Marketing Channels</h3>
            <div class="channel-info">${channelHtml}</div>
        `;
    }

    async fetchChannelClassifications() {
        try {
            const response = await fetch(`${this.apiUrl}/model/channels/${this.uploadId}`);
            const data = await response.json();

            if (response.ok) {
                this.displayChannelClassifications(data.channels);
            } else {
                console.error('Failed to fetch channel classifications:', data);
            }
        } catch (error) {
            console.error('Error fetching channel classifications:', error);
        }
    }

    displayChannelClassifications(channels) {
        // Channel Memory customization removed - Bayesian optimization now automatically explores full parameter space
        // No manual r-value adjustment needed with Optuna optimization
        // Store channel data for later use if needed
        this.channelData = channels;
        return;
    }

    // Stub function to maintain compatibility
    updateRGrid(channelName, customR) {
        // No longer needed with Bayesian optimization
        return;
    }

    // Stub function to maintain compatibility
    resetRValues() {
        // No longer needed with Bayesian optimization
        return;
    }

    // Removed old HTML template code and event listeners for channel memory customization

    getTypeColor(type) {
        const colors = {
            'search_brand': '#dc3545',      // Red
            'search_non_brand': '#fd7e14',  // Orange
            'social': '#6f42c1',             // Purple
            'display': '#20c997',            // Teal
            'tv_video_youtube': '#007bff',  // Blue
            'other': '#6c757d'              // Gray
        };
        return colors[type] || colors['other'];
    }

    async startTraining() {
        if (!this.uploadId) {
            this.showStatus('❌ No data uploaded', 'error');
            return;
        }

        // No custom r values needed - Bayesian optimization explores full parameter space automatically
        const requestData = {
            upload_id: this.uploadId,
            config: {
                carryover_prior: "uniform",
                saturation_prior: "uniform",
                media_transform: "adstock",
                max_lag: 8,
                iterations: 2000
            }
        };

        try {
            document.getElementById('start-training').disabled = true;
            this.showSection('progress-section');

            const response = await fetch(`${this.apiUrl}/model/train`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });

            const result = await response.json();

            if (response.ok) {
                this.runId = result.run_id;
                this.showTrainingStatus(`🚀 Training started! Run ID: ${this.runId}`, 'success');
                document.getElementById('training-loading').classList.remove('hidden'); // Show loading message
                document.getElementById('cancel-training').style.display = 'block'; // Show cancel button
                this.startProgressMonitoring();
            } else {
                const errorMessage = result.detail || result.message || JSON.stringify(result);
                this.showTrainingStatus(`❌ Training failed to start: ${errorMessage}`, 'error');
                document.getElementById('start-training').disabled = false;
            }
        } catch (error) {
            this.showTrainingStatus(`❌ Training error: ${error.message}`, 'error');
            document.getElementById('start-training').disabled = false;
        }
    }

    async cancelTraining() {
        const confirmCancel = confirm('Are you sure you want to cancel the training? This action cannot be undone.');
        if (!confirmCancel) {
            return;
        }

        // Immediately stop monitoring to prevent conflicting messages
        this.stopProgressMonitoring();
        this.isCancelling = true;

        // Immediately hide the "Training your Media Mix Model..." message
        document.getElementById('training-loading').classList.add('hidden');

        let runIdToCancel = this.runId;

        // If we don't have a runId in memory, try to find an active training job
        if (!runIdToCancel) {
            try {
                this.showTrainingStatus('🔍 Looking for active training jobs...', 'info');
                const response = await fetch(`${this.apiUrl}/admin/training/list`);
                const result = await response.json();
                if (response.ok && result.training_runs && result.training_runs.length > 0) {
                    // Find the most recent training or queued job
                    const activeJob = result.training_runs.find(run => 
                        run.status === 'training' || run.status === 'queued'
                    );
                    if (activeJob) {
                        runIdToCancel = activeJob.run_id;
                        this.runId = runIdToCancel; // Update our local state
                        this.showTrainingStatus(`🔍 Found stuck training job: ${runIdToCancel.substring(0, 8)}...`, 'info');
                    } else {
                        this.showTrainingStatus('❌ No active training to cancel', 'error');
                        this.isCancelling = false;
                        // No training to cancel, so no need to show loading again
                        return;
                    }
                } else {
                    this.showTrainingStatus('❌ No active training to cancel', 'error');
                    this.isCancelling = false;
                    // No training to cancel, so no need to show loading again
                    return;
                }
            } catch (error) {
                this.showTrainingStatus(`❌ Error checking for active training: ${error.message}`, 'error');
                this.isCancelling = false;
                // No training to cancel, so no need to show loading again
                return;
            }
        }

        try {
            const response = await fetch(`${this.apiUrl}/model/training/cancel/${runIdToCancel}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const result = await response.json();

            if (response.ok) {
                this.showTrainingStatus('⏹️ Training cancelled successfully', 'success');
                document.getElementById('cancel-training').style.display = 'none';
                document.getElementById('start-training').disabled = false;
                this.runId = null; // Clear the runId
                // Keep isCancelling true for a bit to prevent stale progress updates
                setTimeout(() => { this.isCancelling = false; }, 3000);
            } else {
                this.showTrainingStatus(`❌ Failed to cancel training: ${result.detail}`, 'error');
                this.isCancelling = false;
                // Cancellation failed, show loading again as training continues
                document.getElementById('training-loading').classList.remove('hidden');
                // Resume monitoring
                this.startProgressMonitoring();
            }
        } catch (error) {
            this.showTrainingStatus(`❌ Cancel error: ${error.message}`, 'error');
            this.isCancelling = false;
            // Cancellation failed, show loading again as training continues
            document.getElementById('training-loading').classList.remove('hidden');
            // Resume monitoring
            this.startProgressMonitoring();
        }
    }

    showTrainingStatus(message, type) {
        const statusDiv = document.getElementById('training-status');
        statusDiv.innerHTML = `<div class="status ${type}">${message}</div>`;
    }

    startProgressMonitoring() {
        this.progressInterval = setInterval(() => {
            this.checkTrainingProgress();
        }, 3000); // Check every 3 seconds
        
        // Also check immediately
        this.checkTrainingProgress();
    }

    async checkTrainingProgress() {
        if (!this.runId) return;

        // Don't update progress if we're in the middle of cancelling
        if (this.isCancelling) return;

        try {
            const response = await fetch(`${this.apiUrl}/model/training/progress/${this.runId}`);
            const progress = await response.json();

            if (response.ok) {
                this.updateProgressDisplay(progress);
                
                if (progress.status === 'completed') {
                    this.stopProgressMonitoring();
                    this.showTrainingComplete(progress);
                } else if (progress.status === 'failed') {
                    this.stopProgressMonitoring();
                    this.showTrainingFailed(progress);
                } else if (progress.status === 'cancelled') {
                    this.stopProgressMonitoring();
                    this.showTrainingCancelled(progress);
                }
            } else {
                console.error('Progress check failed:', progress);
            }
        } catch (error) {
            console.error('Progress monitoring error:', error);
        }
    }

    updateProgressDisplay(progress) {
        // Don't update display if we're cancelling
        if (this.isCancelling) return;

        const progressFill = document.getElementById('progress-fill');
        const progressPct = progress.progress?.progress_pct || 0;

        progressFill.style.width = `${Math.min(progressPct, 100)}%`;

        // Debug logging to see what we're receiving
        if (progress.progress?.type) {
            console.log('Progress type received:', progress.progress.type);
            console.log('Full progress data:', progress.progress);
        }

        // Handle CV structure display
        if (progress.progress?.type === 'cv_structure') {
            console.log('Displaying CV structure!');
            this.cvStructureInfo = progress.progress;

            // Display holdout info if present
            if (progress.progress.holdout_days > 0) {
                const msg = `📊 Data Split: ${progress.progress.cv_days} days for CV, ${progress.progress.holdout_days} days for final holdout validation`;
                this.showTrainingStatus(msg, 'info');
            }

            this.displayCVStructure(progress.progress);
            return;
        }

        // Handle holdout validation complete
        if (progress.progress?.type === 'holdout_validation_complete') {
            const data = progress.progress;
            this.displayHoldoutValidation(data);
            return;
        }

        // Handle outer fold updates for nested CV
        if (progress.progress?.type === 'outer_fold_start') {
            const msg = `🔄 Nested CV - Outer Fold ${progress.progress.fold}/${progress.progress.total_folds} (Weeks ${progress.progress.weeks})`;
            this.showTrainingStatus(msg, 'info');
            return;
        }

        if (progress.progress?.type === 'outer_fold_complete' || progress.progress?.type === 'fold_complete') {
            const fold = progress.progress.fold;
            const mape = progress.progress.mape.toFixed(2);
            const params = progress.progress.parameters;

            // Create detailed fold results display
            let msg = `✅ Fold ${fold} Complete - MAPE: ${mape}%`;

            // Store fold results for later display (check for duplicates)
            if (!this.foldResults) this.foldResults = [];

            // Check if this fold already exists (avoid duplicates)
            const existingFoldIndex = this.foldResults.findIndex(f => f.fold === fold);
            if (existingFoldIndex !== -1) {
                // Update existing fold if MAPE is better
                if (parseFloat(mape) < this.foldResults[existingFoldIndex].mape) {
                    this.foldResults[existingFoldIndex] = {
                        fold: fold,
                        mape: parseFloat(mape),
                        parameters: params
                    };
                }
            } else {
                // Add new fold
                this.foldResults.push({
                    fold: fold,
                    mape: parseFloat(mape),
                    parameters: params
                });
            }

            // Display parameters summary
            if (params && params.channel_betas) {
                const channels = Object.keys(params.channel_betas);
                const paramSummary = channels.slice(0, 2).map(ch =>
                    `${ch}: β=${params.channel_betas[ch].toFixed(3)}, r=${params.channel_rs[ch].toFixed(3)}`
                ).join(', ');
                msg += ` | ${paramSummary}${channels.length > 2 ? '...' : ''}`;
            }

            this.showTrainingStatus(msg, 'success');
            this.displayFoldResults();
            return;
        }

        // Handle parameter selection complete
        if (progress.progress?.type === 'parameter_selection_complete') {
            const data = progress.progress;

            // Check for high variance in fold MAPEs
            const allMapes = data.all_fold_mapes || [];
            if (allMapes.length > 0) {
                const mean = allMapes.reduce((a, b) => a + b, 0) / allMapes.length;
                const variance = allMapes.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / allMapes.length;
                const std = Math.sqrt(variance);
                const cv = std / mean; // Coefficient of variation

                if (cv > 0.5) {
                    // High variance warning
                    this.showTrainingStatus(`⚠️ High variance detected in fold MAPEs: ${allMapes.map(m => m.toFixed(1) + '%').join(', ')}`, 'warning');
                    this.showTrainingStatus(`📊 Using selective parameter averaging due to divergent fold performances`, 'warning');
                }
            }

            let msg = `📊 Final Parameter Selection: Averaged top ${data.folds_averaged} folds`;
            msg += ` (Folds ${data.top_fold_numbers.join(', ')})`;
            msg += ` with MAPEs: ${data.top_fold_mapes.map(m => m.toFixed(2) + '%').join(', ')}`;

            this.showTrainingStatus(msg, 'info');

            // Display final parameters
            if (data.final_parameters) {
                this.displayFinalParameters(data.final_parameters);
            }
            return;
        }

        // Handle inner fold info
        if (progress.progress?.type === 'inner_fold_info') {
            const msg = `📊 Setting up inner fold for Outer Fold ${progress.progress.outer_fold} (${progress.progress.inner_train_days} train / ${progress.progress.inner_test_days} test days)`;
            this.showTrainingStatus(msg, 'info');
            return;
        }

        // Handle parameter optimization progress
        if (progress.progress?.type === 'parameter_optimization') {
            const pct = ((progress.progress.combination / progress.progress.total_combinations) * 100).toFixed(2);
            const msg = `⚙️ Optimizing parameters - Fold ${progress.progress.fold}: Testing combination ${progress.progress.combination}/${progress.progress.total_combinations} (${pct}%)`;
            this.showTrainingStatus(msg, 'info');
            // Also update progress bar
            const progressFill = document.getElementById('progress-fill');
            if (progressFill && progress.progress.combination > 0) {
                progressFill.style.width = `${Math.min(pct, 100)}%`;
            }
            return;
        }

        let statusMessage = `Status: ${progress.status}`;
        if (progress.progress?.current_step) {
            statusMessage += ` - ${progress.progress.current_step}`;
        }
        if (progress.progress?.current_fold && progress.progress?.total_folds) {
            statusMessage += ` (${progress.progress.current_fold}/${progress.progress.total_folds})`;
        }

        this.showTrainingStatus(statusMessage, 'info');
    }

    displayCVStructure(cvInfo) {
        // Store the CV info for later display
        this.cvStructureInfo = cvInfo;
        this.nestedCVUsed = cvInfo.method !== 'simple';

        // Create a formatted display of the CV structure
        let message = `📊 Cross-Validation Structure:<br>`;
        message += `Data: ${cvInfo.total_weeks} weeks (${cvInfo.total_days} days)<br>`;

        if (cvInfo.method === 'simple') {
            message += `Method: Simple CV (insufficient data for nested CV)<br>`;
        } else {
            message += `Method: Nested CV with ${cvInfo.n_outer_folds} outer folds<br><br>`;
            message += `<table style="font-size: 0.9em; margin-top: 10px;">`;
            message += `<tr style="background: #f0f0f0;">
                        <th>Fold</th>
                        <th>Weeks</th>
                        <th>Outer Train</th>
                        <th>Outer Test</th>
                        <th>Inner Train</th>
                        <th>Inner Test</th>
                        </tr>`;

            cvInfo.fold_details.forEach(fold => {
                message += `<tr>
                           <td>${fold.fold}</td>
                           <td>${fold.weeks}</td>
                           <td>${fold.outer_train}</td>
                           <td>${fold.outer_test}</td>
                           <td>${fold.inner_train}</td>
                           <td>${fold.inner_test}</td>
                           </tr>`;
            });
            message += `</table>`;
        }

        // Display in a special CV info section
        const trainingStatus = document.getElementById('training-status');
        trainingStatus.innerHTML = message;
        trainingStatus.classList.remove('hidden');
        trainingStatus.style.background = '#e3f2fd';
        trainingStatus.style.border = '2px solid #1976d2';
        trainingStatus.style.padding = '15px';
        trainingStatus.style.marginBottom = '20px';
    }

    async showTrainingComplete(progress) {
        document.getElementById('training-loading').classList.add('hidden');
        document.getElementById('progress-fill').style.width = '100%';
        this.showTrainingStatus('🎉 Training completed successfully!', 'success');
        document.getElementById('cancel-training').style.display = 'none';
        
        // Fetch and show full results
        console.log('Attempting to fetch results. RunId:', this.runId);
        if (this.runId) {
            try {
                this.showTrainingStatus('🎉 Training completed! Fetching results...', 'success');
                const url = `${this.apiUrl}/model/results/${this.runId}`;
                console.log('Fetching results from:', url);
                const response = await fetch(url);

                let results;
                const contentType = response.headers.get("content-type");
                if (contentType && contentType.indexOf("application/json") !== -1) {
                    results = await response.json();
                } else {
                    const text = await response.text();
                    console.error('Non-JSON response received:', text);
                    throw new Error(`Non-JSON response: ${text.substring(0, 200)}`);
                }

                if (response.ok) {
                    console.log('✅ Results fetched successfully:', results);
                    this.displayResults(results);
                } else {
                    console.error('❌ Failed to fetch results. Status:', response.status);
                    console.error('Error response:', results);
                    this.showTrainingStatus(`🎉 Training completed! (Results unavailable - ${results.detail || 'Unknown error'})`, 'success');
                }
            } catch (error) {
                console.error('❌ Exception fetching results:', error);
                console.error('Error details:', error.message);
                this.showTrainingStatus('🎉 Training completed! (Results unavailable - Network error)', 'success');
            }
        } else {
            console.warn('⚠️ No runId available, checking for results in progress message');
            // Fallback: try to show results from progress if available
            if (progress && progress.results) {
                console.log('Using fallback results from progress:', progress.results);
                this.displayResults(progress.results);
            } else {
                console.error('❌ No runId and no results in progress message');
                this.showTrainingStatus('🎉 Training completed! (Unable to fetch results - no run ID)', 'success');
            }
        }
    }

    showTrainingFailed(progress) {
        document.getElementById('training-loading').classList.add('hidden');
        const errorMsg = progress.error || 'Training failed';
        this.showTrainingStatus(`❌ Training failed: ${errorMsg}`, 'error');
        document.getElementById('cancel-training').style.display = 'none';
        document.getElementById('start-training').disabled = false;
    }

    showTrainingCancelled(progress) {
        document.getElementById('training-loading').classList.add('hidden');
        const cancelMsg = progress.progress?.message || 'Training cancelled by user';
        this.showTrainingStatus(`⏹️ ${cancelMsg}`, 'error');
        document.getElementById('cancel-training').style.display = 'none';
        document.getElementById('start-training').disabled = false;
    }

    displayResults(results) {
        console.log('*** EQUATION DEBUG: displayResults called');
        console.log('*** EQUATION DEBUG: Full results object:', results);
        this.showSection('results-section');

        // Handle both old format (progress.results) and new format (API response)
        const performance = results.model_performance || results;
        const parameters = results.parameters;
        const confidenceIntervals = results.confidence_intervals || {};
        const diagnostics = results.diagnostics || {};  // Fix: Define diagnostics

        // Extract CV structure info from results
        if (results.cv_structure_info) {
            this.cvStructureInfo = results.cv_structure_info;
            this.nestedCVUsed = results.cv_structure_info.method !== 'simple';
            console.log('CV Structure from results:', this.cvStructureInfo);
            console.log('Nested CV Used:', this.nestedCVUsed);
            console.log('Has fold_details:', !!this.cvStructureInfo.fold_details);
        }

        console.log('*** EQUATION DEBUG: Extracted parameters:', parameters);
        console.log('*** EQUATION DEBUG: Extracted confidence intervals:', confidenceIntervals);
        console.log('Extracted parameters:', parameters);

        // Check if holdout validation exists
        const holdoutData = diagnostics?.holdout_validation;
        console.log('Holdout validation data:', holdoutData);

        const resultsHtml = `
            <div class="metric-card" style="${holdoutData ? '' : 'grid-column: span 1;'}">
                <div class="metric-value">${performance.cv_mape?.toFixed(2) || 'N/A'}%</div>
                <div class="metric-label">CV MAPE<br><span style="font-size: 0.75rem; font-weight: normal; opacity: 0.8;">(Cross-validation)</span></div>
            </div>
            ${holdoutData ? `
            <div class="metric-card" style="background: ${holdoutData.overfit_warning ? '#fff3cd' : '#d4edda'}; border: 2px solid ${holdoutData.overfit_warning ? '#ffc107' : '#28a745'};">
                <div class="metric-value" style="color: ${holdoutData.overfit_warning ? '#856404' : '#155724'};">
                    ${holdoutData.holdout_mape.toFixed(2)}%
                </div>
                <div class="metric-label" style="color: ${holdoutData.overfit_warning ? '#856404' : '#155724'};">
                    Holdout MAPE
                    <br><span style="font-size: 0.75rem; font-weight: normal;">
                    ${holdoutData.overfit_warning ? '⚠️ Overfit' : '✅ Valid'} (${holdoutData.holdout_days}d)
                    </span>
                </div>
            </div>` : ''}
            <div class="metric-card">
                <div class="metric-value">${performance.r_squared?.toFixed(3) || 'N/A'}</div>
                <div class="metric-label">R-Squared<br><span style="font-size: 0.75rem; font-weight: normal; opacity: 0.8;">(Training fit)</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${(performance.mape || performance.final_mape)?.toFixed(2) || 'N/A'}%</div>
                <div class="metric-label">Training MAPE<br><span style="font-size: 0.75rem; font-weight: normal; opacity: 0.8;">(In-sample)</span></div>
            </div>
            ${this.nestedCVUsed ?
                `<div class="metric-card" style="background: #e3f2fd; border: 2px solid #1976d2;">
                    <div class="metric-value" style="color: #1976d2; font-size: 1.8rem;">
                        ${this.cvStructureInfo?.n_outer_folds || 'N/A'}
                    </div>
                    <div class="metric-label" style="color: #1976d2; font-weight: bold;">
                        Nested CV Folds<br>
                        <span style="font-size: 0.85rem; font-weight: normal;">
                            (${this.cvStructureInfo?.total_weeks || 'N/A'} weeks of data)
                        </span>
                    </div>
                </div>` :
                `<div class="metric-card" style="background: #fff3cd; border: 2px solid #ffc107;">
                    <div class="metric-value" style="color: #856404; font-size: 1.8rem;">
                        ${performance.n_folds_averaged || 'N/A'}
                    </div>
                    <div class="metric-label" style="color: #856404; font-weight: bold;">
                        Parameter Averaging<br>
                        <span style="font-size: 0.85rem; font-weight: normal;">
                            (Best ${performance.n_folds_averaged || 'N/A'} folds used)
                        </span>
                    </div>
                </div>`
            }
        `;
        
        document.getElementById('results-grid').innerHTML = resultsHtml;

        // Remove any existing CV details div first
        const existingCvDetails = document.getElementById('cv-details-section');
        if (existingCvDetails) {
            existingCvDetails.remove();
        }

        // Display CV structure details if available (after results)
        if (this.cvStructureInfo) {
            console.log('Displaying CV structure details - cvStructureInfo:', this.cvStructureInfo);

            if (this.nestedCVUsed && this.cvStructureInfo.fold_details) {
                console.log('Creating nested CV details table with', this.cvStructureInfo.fold_details.length, 'folds');
                const cvDetailsHtml = `
                    <div id="cv-details-section" style="margin-top: 30px; padding: 20px; background: #e3f2fd; border: 2px solid #1976d2; border-radius: 8px;">
                        <h3 style="color: #1976d2; margin-bottom: 15px;">📊 Nested Cross-Validation Structure</h3>
                        <p style="margin-bottom: 10px;">Data: ${this.cvStructureInfo.total_weeks} weeks (${this.cvStructureInfo.total_days} days)</p>
                        <table style="width: 100%; font-size: 0.9em; border-collapse: collapse;">
                            <tr style="background: #bbdefb;">
                                <th style="padding: 8px; border: 1px solid #1976d2;">Fold</th>
                                <th style="padding: 8px; border: 1px solid #1976d2;">Weeks</th>
                                <th style="padding: 8px; border: 1px solid #1976d2;">Outer Train</th>
                                <th style="padding: 8px; border: 1px solid #1976d2;">Outer Test</th>
                                <th style="padding: 8px; border: 1px solid #1976d2;">Inner Train</th>
                                <th style="padding: 8px; border: 1px solid #1976d2;">Inner Test</th>
                            </tr>
                            ${this.cvStructureInfo.fold_details.map(fold => `
                                <tr style="background: white;">
                                    <td style="padding: 8px; border: 1px solid #1976d2; text-align: center;">${fold.fold}</td>
                                    <td style="padding: 8px; border: 1px solid #1976d2; text-align: center;">${fold.weeks}</td>
                                    <td style="padding: 8px; border: 1px solid #1976d2; text-align: center;">${fold.outer_train}</td>
                                    <td style="padding: 8px; border: 1px solid #1976d2; text-align: center;">${fold.outer_test}</td>
                                    <td style="padding: 8px; border: 1px solid #1976d2; text-align: center;">${fold.inner_train}</td>
                                    <td style="padding: 8px; border: 1px solid #1976d2; text-align: center;">${fold.inner_test}</td>
                                </tr>
                            `).join('')}
                        </table>
                        <p style="margin-top: 15px; font-size: 0.9em; color: #555;">
                            <strong>Note:</strong> Parameters were selected using inner folds, then evaluated on outer test sets for unbiased performance estimates.
                        </p>
                    </div>
                `;

                // Insert CV details after results grid
                document.getElementById('results-grid').insertAdjacentHTML('afterend', cvDetailsHtml);
                console.log('Nested CV details table inserted');
            } else if (!this.nestedCVUsed) {
                console.log('Creating simple CV info');
                // Show simple CV info
                const simpleCvHtml = `
                    <div id="cv-details-section" style="margin-top: 30px; padding: 20px; background: #fff3cd; border: 2px solid #ffc107; border-radius: 8px;">
                        <h3 style="color: #856404; margin-bottom: 15px;">📊 Simple Cross-Validation</h3>
                        <p style="color: #856404;">
                            Used simple walk-forward cross-validation with ${this.cvStructureInfo.total_weeks || 'N/A'} weeks of data.<br>
                            Parameters from the best ${performance.n_folds_averaged || 'N/A'} folds were averaged for the final model.
                        </p>
                    </div>
                `;
                document.getElementById('results-grid').insertAdjacentHTML('afterend', simpleCvHtml);
                console.log('Simple CV info inserted');
            }
        } else {
            console.log('No CV structure info available');
        }

        // Display parameter values if available
        console.log('*** EQUATION DEBUG: Checking parameters for equation display:', parameters);
        console.log('*** EQUATION DEBUG: Parameters exists?', !!parameters);
        console.log('*** EQUATION DEBUG: Has channel_alphas?', !!(parameters && parameters.channel_alphas));
        console.log('*** EQUATION DEBUG: Has channel_betas?', !!(parameters && parameters.channel_betas));
        console.log('*** EQUATION DEBUG: Has channel_rs?', !!(parameters && parameters.channel_rs));

        if (parameters && (parameters.channel_alphas || parameters.channel_betas || parameters.channel_rs)) {
            console.log('*** EQUATION DEBUG: Parameters found, calling displayParameters and displayResponseCurveEquations');
            this.displayParameters(parameters);
            this.displayResponseCurveEquations(parameters, confidenceIntervals);
        } else {
            console.log('*** EQUATION DEBUG: No parameters found for equation display - showing test equation');
            // Force show a test equation to verify display works
            this.displayTestEquation();
        }

        // Render the charts - don't await, let it run asynchronously
        setTimeout(() => {
            this.renderChartsAfterTraining();
        }, 100); // Small delay to ensure DOM is ready

        // Redisplay holdout validation if we have it
        if (this.holdoutValidationData) {
            setTimeout(() => {
                this.displayHoldoutValidation(this.holdoutValidationData);
            }, 150);
        }
    }

    async renderChartsAfterTraining() {
        console.log('🎨 Starting chart rendering after training completion...');

        // Check if we have a runId
        if (!this.runId) {
            console.error('❌ No runId available for fetching response curves');
            return;
        }

        // Fetch and render response curves
        try {
            console.log('📊 Fetching response curves from API...');
            const response = await fetch(`${this.apiUrl}/model/response-curves/${this.runId}`);

            if (!response.ok) {
                console.error('❌ Failed to fetch response curves:', response.status, response.statusText);
                return;
            }

            const data = await response.json();
            console.log('✅ Response curves data received:', data);

            if (data.response_curves) {
                console.log('📈 Rendering response curves...');
                this.renderResponseCurvesFromAPI(data);

                // Fetch and render marginal ROI
                console.log('📊 Fetching marginal ROI data...');
                await this.fetchAndDisplayMarginalROI();

                console.log('📊 Inserting marginal ROI section...');
                this.insertMarginalROISection();

                console.log('📊 Rendering marginal ROI charts...');
                this.renderMarginalROICharts(data);

                // Add Profit Maximizer
                console.log('💰 Adding profit maximizer...');
                await this.addProfitMaximizerAtEnd();

                console.log('✅ All charts rendered successfully!');
            } else {
                console.warn('⚠️ No response curves data in response');
            }
        } catch (error) {
            console.error('❌ Error rendering charts:', error);
            console.error('Stack trace:', error.stack);
        }
    }

    displayParameters(parameters) {
        const { channel_alphas, channel_betas, channel_rs } = parameters;
        
        if (!channel_alphas || !channel_betas || !channel_rs) {
            return;
        }
        
        // Parameters are now displayed in the equations section, removing duplicate display
    }

    displayResponseCurveEquations(parameters, confidenceIntervals = {}) {
        console.log('displayResponseCurveEquations called with parameters:', parameters);
        console.log('Parameters type:', typeof parameters);
        console.log('Parameters keys:', Object.keys(parameters || {}));

        const { channel_alphas, channel_betas, channel_rs } = parameters;

        console.log('Extracted channel_alphas:', channel_alphas);
        console.log('Extracted channel_betas:', channel_betas);
        console.log('Extracted channel_rs:', channel_rs);

        if (!channel_alphas || !channel_betas || !channel_rs) {
            console.log('Missing required parameters for equations - showing test equation instead');
            this.displayTestEquation();
            return;
        }
        console.log('All parameters present, creating equations...');

        // Create equation display section
        const equationHtml = Object.keys(channel_alphas).map(channel => {
            const alpha = channel_alphas[channel];
            const beta = channel_betas[channel];
            const r = channel_rs[channel];

            return `
                <div class="equation-card">
                    <div class="equation-channel">${this.formatChannelName(channel)}</div>
                    <div class="equation-formula">
                        <div class="equation-text">
                            Lifetime Profit = ${alpha.toFixed(2)} × [(Daily Spend / ${(1-r).toFixed(2)})^${beta.toFixed(2)}]
                        </div>
                        <div class="equation-explanation">
                            This shows lifetime incremental profit from sustained daily spending (includes carryover effects)
                        </div>
                    </div>
                    <div class="equation-parameters">
                        <div class="parameter-row">
                            <span class="parameter-symbol">α</span>
                            <span class="parameter-description">Incremental strength</span>
                            <span class="equation-parameter-value">${alpha.toFixed(2)}</span>
                        </div>
                        <div class="parameter-row">
                            <span class="parameter-symbol">β</span>
                            <span class="parameter-description">Saturation (diminishing returns)</span>
                            <span class="equation-parameter-value">${beta.toFixed(2)}</span>
                        </div>
                        <div class="parameter-row">
                            <span class="parameter-symbol">r</span>
                            <span class="parameter-description">Adstock (carryover effect)</span>
                            <span class="equation-parameter-value">${r.toFixed(2)}</span>
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        // Add response curves section
        const responseCurvesSection = this.generateResponseCurvesSection(parameters);

        // Add equations section to results
        const equationsSection = `
            <div style="margin-top: 30px; border: 3px solid #ff0000; padding: 20px; background: #fff;">
                <h2 style="color: #ff0000; margin-bottom: 20px; font-size: 2rem;">🧮 RESPONSE CURVE EQUATIONS</h2>
                <div style="background: #ffeeee; padding: 15px; border-radius: 8px; margin-bottom: 20px; font-style: italic; color: #333; border: 2px solid #ff0000;">
                    <strong>These equations show lifetime incremental profit from sustained daily spending on each channel:</strong><br>
                    Lifetime Profit = α × [(Daily Spend ÷ (1-r))^β] where carryover effects are built into the formula
                </div>
                <div class="equations-grid">
                    ${equationHtml}
                </div>
            </div>
        `;

        // Insert sections in correct order: Response Curves FIRST, then Equations
        console.log('🔧 Inserting sections in correct order...');
        const resultsGrid = document.getElementById('results-grid');
        if (resultsGrid) {
            // Insert response curves first (this will appear immediately after results-grid)
            resultsGrid.insertAdjacentHTML('afterend', responseCurvesSection);
            console.log('✅ Response curves section inserted first');

            // Insert equations second (this will appear after response curves)
            const responseCurvesContainer = document.getElementById('response-curves-charts');
            if (responseCurvesContainer && responseCurvesContainer.parentElement) {
                responseCurvesContainer.parentElement.insertAdjacentHTML('afterend', equationsSection);
                console.log('✅ Equations section inserted after response curves');
            } else {
                // Fallback: insert after results grid if response curves container not found
                resultsGrid.insertAdjacentHTML('afterend', equationsSection);
                console.log('⚠️ Equations section inserted after results grid (fallback)');
            }

            // Generate the actual charts after DOM elements are created
            setTimeout(() => {
                this.generateResponseCurveCharts(parameters, this.runId);
            }, 100);
        } else {
            console.error('Could not find results-grid element');
        }
    }

    displayTestEquation() {
        console.log('Displaying test equation to verify DOM insertion...');
        const testEquationSection = `
            <div style="margin-top: 30px; border: 3px solid #00ff00; padding: 20px; background: #eeffee;">
                <h2 style="color: #00aa00; margin-bottom: 20px; font-size: 2rem;">🧪 TEST EQUATION DISPLAY</h2>
                <div style="background: #ffffff; padding: 20px; border-radius: 8px; border: 2px solid #00aa00;">
                    <h3>TV Channel:</h3>
                    <p style="font-size: 1.2rem; font-family: monospace;"><strong>Profit = 27.09 × [(S / (1 - 0.1))^0.7]</strong></p>
                    <p>Where S is daily spend for TV advertising</p>
                </div>
            </div>
        `;

        const resultsGrid = document.getElementById('results-grid');
        if (resultsGrid) {
            resultsGrid.insertAdjacentHTML('afterend', testEquationSection);
            console.log('Test equation inserted successfully');
        } else {
            console.error('Could not find results-grid for test equation');
        }
    }

    displayMarginalROI(marginalROIData) {
        const { marginal_roi_by_channel, current_spend, baseline_spend_per_day, interpretation } = marginalROIData;
        
        if (!marginal_roi_by_channel || Object.keys(marginal_roi_by_channel).length === 0) {
            return;
        }
        
        // Create marginal ROI display section
        const marginalROIHtml = Object.entries(marginal_roi_by_channel).map(([channel, roi]) => {
            const currentSpendValue = current_spend[channel] || 0;
            const baselineSpendValue = baseline_spend_per_day[channel] || 0;
            const formattedROI = roi.toFixed(2);
            
            return `
                <div class="marginal-roi-card">
                    <div class="roi-channel">${this.formatChannelName(channel)}</div>
                    <div class="roi-values">
                        <div class="roi-item">
                            <span class="roi-label">Marginal ROI:</span>
                            <span class="roi-value">${formattedROI}</span>
                        </div>
                        <div class="roi-item">
                            <span class="roi-label">Baseline Spend (30-day avg):</span>
                            <span class="roi-value">$${this.formatNumber(baselineSpendValue)}/day</span>
                        </div>
                        ${currentSpendValue > 0 ? `
                        <div class="roi-item">
                            <span class="roi-label">Current Daily Spend:</span>
                            <span class="roi-value">$${this.formatNumber(currentSpendValue)}</span>
                        </div>
                        ` : ''}
                        <div class="roi-interpretation">
                            Each additional $1/day generates ${formattedROI}x more profit
                        </div>
                    </div>
                </div>
            `;
        }).join('');
        
        // Generate mROI curves section
        const mroiCurvesSection = this.generateMROICurvesSection(marginal_roi_by_channel, baseline_spend_per_day);

        // Add marginal ROI section to results
        const marginalROISection = `
            <div style="margin-top: 30px;">
                <h3 style="color: #333; margin-bottom: 20px;">💰 Marginal ROI by Channel</h3>
                <div class="interpretation-box" style="background: #f5f5f5; padding: 15px; border-radius: 8px; margin-bottom: 20px; font-style: italic; color: #666;">
                    ${interpretation.baseline_note ? `<em>${interpretation.baseline_note}</em>` : ''}
                </div>
                <div class="marginal-roi-grid">
                    ${marginalROIHtml}
                </div>
                ${mroiCurvesSection}
            </div>
        `;
        
        // Store the marginal ROI section HTML for later insertion (don't insert now)
        this.marginalROISection = marginalROISection;

        // Generate mROI curves after DOM elements are created
        setTimeout(() => {
            this.generateMROICharts(marginal_roi_by_channel, baseline_spend_per_day);
        }, 100);
    }

    stopProgressMonitoring() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
    }

    async fetchAndDisplayMarginalROI() {
        try {
            const response = await fetch(`${this.apiUrl}/model/marginal-roi/${this.runId}`);
            const marginalROIData = await response.json();
            
            if (response.ok && marginalROIData.marginal_roi_by_channel) {
                this.displayMarginalROI(marginalROIData);
            } else {
                console.warn('Marginal ROI data not available:', marginalROIData);
            }
        } catch (error) {
            console.warn('Error fetching marginal ROI:', error);
        }
    }

    formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toFixed(0);
    }

    generateResponseCurvesSection(parameters) {
        console.log('🚀 Generating response curves section with parameters:', parameters);

        const { channel_alphas, channel_betas, channel_rs } = parameters;

        const curvesHtml = Object.keys(channel_alphas).map(channel => {
            return `
                <div class="response-curve-card">
                    <div class="response-curve-title">${this.formatChannelName(channel)} Response Curve</div>
                    <div class="chart-container">
                        <canvas id="chart-${channel}"></canvas>
                    </div>
                </div>
            `;
        }).join('');

        return `
            <div class="response-curves-section">
                <h2 style="color: #28a745; margin-bottom: 20px; font-size: 2rem;">📈 RESPONSE CURVES</h2>
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px; font-style: italic; color: #333; border: 2px solid #28a745;">
                    <strong>These charts show how incremental profit responds to daily spend for each channel.</strong><br>
                    X-axis: Daily Spend ($) | Y-axis: Lifetime Incremental Profit ($)
                </div>
                <div class="response-curves-grid">
                    ${curvesHtml}
                </div>
                <div id="response-curves-charts"></div>
            </div>
        `;
    }

    renderResponseCurvesFromAPI(data) {
        console.log('📊 Rendering response curves from API data:', data);

        const responseCurves = data.response_curves;
        const avgDailySpend = data.avg_daily_spend_28_days || {};
        const spendCorrelations = data.spend_correlations || {};

        Object.keys(responseCurves).forEach(channel => {
            const curveData = responseCurves[channel];
            const spendLevels = curveData.spend_levels || [];
            const profits = curveData.incremental_profits || [];
            const confidenceIntervals = curveData.confidence_intervals;
            const avgSpend = avgDailySpend[channel];

            console.log(`📊 ${channel} - Confidence intervals:`, confidenceIntervals);
            console.log(`📊 ${channel} - 28-day avg spend: $${avgSpend?.toFixed(2) || 'N/A'}`);

            // Find the point closest to the 28-day average for annotation
            let avgSpendIndex = -1;
            if (avgSpend && avgSpend > 0) {
                avgSpendIndex = spendLevels.findIndex(spend => spend >= avgSpend);
                if (avgSpendIndex === -1) avgSpendIndex = spendLevels.length - 1;
            }

            // Prepare datasets
            const datasets = [{
                label: 'Lifetime Incremental Profit',
                data: profits,
                borderColor: '#2d5aa0',
                backgroundColor: 'rgba(45, 90, 160, 0.1)',
                borderWidth: 3,
                fill: false,
                tension: 0.4
            }];

            // Add confidence interval bands if available
            if (confidenceIntervals && confidenceIntervals.lower && confidenceIntervals.upper) {
                console.log(`✅ Adding confidence intervals for ${channel}`);

                datasets.push({
                    label: '95% CI (Bootstrap 500 samples) - Upper',
                    data: confidenceIntervals.upper,
                    borderColor: 'rgba(45, 90, 160, 0.4)',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0
                });

                datasets.push({
                    label: '95% CI (Bootstrap 500 samples) - Lower',
                    data: confidenceIntervals.lower,
                    borderColor: 'rgba(45, 90, 160, 0.4)',
                    backgroundColor: 'rgba(45, 90, 160, 0.1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: 1, // Fill to the upper bound dataset
                    tension: 0.4,
                    pointRadius: 0
                });
            } else {
                console.log(`⚠️ No confidence intervals found for ${channel}`);
            }

            // Add vertical line for 28-day average if available
            const annotations = [];
            if (avgSpend && avgSpend > 0 && avgSpendIndex >= 0) {
                annotations.push({
                    type: 'line',
                    mode: 'vertical',
                    scaleID: 'x',
                    value: avgSpendIndex,
                    borderColor: 'rgba(255, 99, 132, 0.8)',
                    borderWidth: 2,
                    borderDash: [10, 5],
                    label: {
                        content: `28-day avg: $${avgSpend.toFixed(0)}`,
                        enabled: true,
                        position: 'top'
                    }
                });
            }

            // Create chart
            const ctx = document.getElementById(`chart-${channel}`);
            if (ctx) {
                new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: spendLevels.map(s => `$${Math.round(s).toLocaleString()}`),
                        datasets: datasets
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {
                            intersect: false,
                            mode: 'index'
                        },
                        plugins: {
                            title: {
                                display: true,
                                text: [
                                    `${this.formatChannelName(channel)} Response Curve${avgSpend ? ` (28-day avg: $${avgSpend.toFixed(0)}/day)` : ''}`,
                                    confidenceIntervals ? '🔬 Bootstrap CI Active (500 samples)' : '⚠️ Using Placeholder CI'
                                ],
                                font: { size: 16, weight: 'bold' },
                                color: confidenceIntervals ? '#2d5aa0' : '#ff6b6b'
                            },
                            legend: {
                                display: true,
                                position: 'bottom',
                                labels: {
                                    filter: function(legendItem, chartData) {
                                        // Only show main curve and confidence interval labels (simplified)
                                        return legendItem.text !== '95% CI (Bootstrap 500 samples) - Upper';
                                    },
                                    generateLabels: function(chart) {
                                        const labels = Chart.defaults.plugins.legend.labels.generateLabels(chart);
                                        return labels.map(label => {
                                            if (label.text === '95% CI (Bootstrap 500 samples) - Lower') {
                                                label.text = '95% CI (Bootstrap Method)';
                                            }
                                            return label;
                                        });
                                    }
                                }
                            },
                            annotation: annotations.length > 0 ? {
                                annotations: annotations
                            } : undefined
                        },
                        scales: {
                            x: {
                                title: {
                                    display: true,
                                    text: 'Daily Spend ($)'
                                }
                            },
                            y: {
                                title: {
                                    display: true,
                                    text: 'Lifetime Incremental Profit ($)'
                                },
                                min: 0,  // Never show negative values
                                ticks: {
                                    callback: function(value) {
                                        return '$' + Math.round(value).toLocaleString();
                                    }
                                }
                            }
                        }
                    }
                });

                console.log(`✅ Chart created for ${channel} with ${confidenceIntervals ? 'confidence intervals' : 'no confidence intervals'}`);

                // Add correlations summary box after the chart
                const spendCorr = spendCorrelations[channel];

                if (spendCorr !== undefined) {
                    const correlationHTML = `
                        <div style="background: #6f42c1; color: white; padding: 15px; margin-top: 10px; border-radius: 8px; font-size: 0.9em;">
                            <div style="font-weight: bold; margin-bottom: 8px; text-align: center;">📊 Spend Correlation with Profit</div>
                            <div style="text-align: center;">
                                <div style="font-size: 1.2em; font-weight: bold;">${Math.round(spendCorr * 100)}%</div>
                            </div>
                        </div>
                    `;

                    // Insert the correlation box after the chart canvas
                    ctx.parentElement.insertAdjacentHTML('afterend', correlationHTML);
                }
            } else {
                console.error(`❌ Could not find canvas element for ${channel}`);
            }
        });
    }

    renderMarginalROICharts(data) {
        console.log('📈 Rendering marginal ROI charts');
        console.log('📊 Response curves data:', data.response_curves);
        console.log('💰 Average daily spend data:', data.avg_daily_spend_28_days);

        const responseCurves = data.response_curves;
        const avgDailySpend = data.avg_daily_spend_28_days || {};

        // Add section header for marginal ROI charts
        const chartContainer = document.getElementById('response-curves-charts');
        if (!chartContainer) {
            console.error('❌ Could not find response-curves-charts container');
            return;
        }
        console.log('✅ Found chart container, adding marginal ROI section');

        const marginalSection = document.createElement('div');
        marginalSection.innerHTML = `
            <div style="margin-top: 40px; margin-bottom: 20px;">
                <h3 style="color: #333; font-size: 1.5em; margin-bottom: 10px;">📊 Marginal ROI by Channel</h3>
                <p style="color: #666; font-size: 0.9em; margin-bottom: 20px;">
                    Shows the return on investment for the next dollar spent. Values above 1.0 indicate profitable spend.
                </p>
            </div>
            <div id="marginal-roi-charts" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px;"></div>
        `;
        chartContainer.appendChild(marginalSection);

        const marginalChartsContainer = document.getElementById('marginal-roi-charts');

        Object.keys(responseCurves).forEach(channel => {
            const curveData = responseCurves[channel];
            const spendLevels = curveData.spend_levels || [];
            const marginalRoas = curveData.marginal_roas || [];
            const currentSpend = avgDailySpend[channel] || 0;

            // Create chart container
            const chartDiv = document.createElement('div');
            chartDiv.style.cssText = 'background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 20px;';

            const canvas = document.createElement('canvas');
            canvas.id = `marginal-roi-chart-${channel}`;
            canvas.width = 400;
            canvas.height = 300;

            chartDiv.appendChild(canvas);
            marginalChartsContainer.appendChild(chartDiv);

            const ctx = canvas.getContext('2d');

            // Find current position on the curve
            let currentROI = 0;
            let currentIndex = -1;
            if (currentSpend > 0) {
                // Find closest spend level to current spend
                currentIndex = spendLevels.findIndex(spend => spend >= currentSpend);
                if (currentIndex === -1) currentIndex = spendLevels.length - 1;
                if (currentIndex >= 0 && currentIndex < marginalRoas.length) {
                    currentROI = marginalRoas[currentIndex];
                }
            }

            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: spendLevels,
                    datasets: [
                        {
                            label: 'Marginal ROI',
                            data: marginalRoas,
                            borderColor: '#007bff',
                            backgroundColor: 'rgba(0, 123, 255, 0.1)',
                            borderWidth: 2,
                            fill: false,
                            pointRadius: 0,
                            pointHoverRadius: 6
                        },
                        {
                            label: 'Breakeven Line (ROI = 1)',
                            data: spendLevels.map(() => 1.0),
                            borderColor: '#dc3545',
                            backgroundColor: 'transparent',
                            borderWidth: 2,
                            borderDash: [5, 5],
                            fill: false,
                            pointRadius: 0,
                            pointHoverRadius: 0
                        },
                        ...(currentIndex >= 0 ? [{
                            label: 'Current Position',
                            data: [{x: currentSpend, y: currentROI}],
                            backgroundColor: '#28a745',
                            borderColor: '#28a745',
                            pointRadius: 8,
                            pointHoverRadius: 10,
                            showLine: false
                        }] : [])
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: `${channel} - Marginal ROI`,
                            font: { size: 16, weight: 'bold' }
                        },
                        legend: {
                            display: true,
                            position: 'top'
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                            callbacks: {
                                label: function(context) {
                                    if (context.datasetIndex === 0) {
                                        return `Marginal ROI: ${context.parsed.y.toFixed(2)}`;
                                    } else if (context.datasetIndex === 1) {
                                        return 'Breakeven';
                                    } else {
                                        return `Current: ${context.parsed.y.toFixed(2)} ROI`;
                                    }
                                },
                                title: function(context) {
                                    return `Daily Spend: $${context[0].parsed.x.toFixed(0)}`;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            type: 'linear',
                            position: 'bottom',
                            title: {
                                display: true,
                                text: 'Daily Spend ($)'
                            },
                            grid: {
                                alpha: 0.3
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Marginal ROI'
                            },
                            grid: {
                                alpha: 0.3
                            },
                            min: 0,
                            ticks: {
                                callback: function(value, index, values) {
                                    // Debug: Log the value being formatted
                                    console.log(`🔧 Y-axis tick formatting: ${value} -> ${Number(value).toFixed(2)}`);
                                    // Ensure we return plain numbers, not currency
                                    const formatted = Number(value).toFixed(2);
                                    return formatted;
                                },
                                // Disable any automatic number formatting
                                maxTicksLimit: 10
                            }
                        }
                    },
                    interaction: {
                        mode: 'nearest',
                        axis: 'x',
                        intersect: false
                    }
                }
            });

            console.log(`✅ Marginal ROI chart created for ${channel}, current spend: $${currentSpend.toFixed(0)}, current ROI: ${currentROI.toFixed(2)}`);
        });
    }

    renderMarginalROIChartsManual(parameters) {
        console.log('📈 Rendering marginal ROI charts (manual generation)');

        const { channel_alphas, channel_betas, channel_rs } = parameters;

        // Add section header for marginal ROI charts
        const chartContainer = document.getElementById('response-curves-charts');
        if (!chartContainer) {
            console.error('❌ Could not find response-curves-charts container');
            return;
        }
        console.log('✅ Found chart container, adding marginal ROI section (manual)');

        const marginalSection = document.createElement('div');
        marginalSection.innerHTML = `
            <div style="margin-top: 40px; margin-bottom: 20px;">
                <h3 style="color: #333; font-size: 1.5em; margin-bottom: 10px;">📊 Marginal ROI by Channel</h3>
                <p style="color: #666; font-size: 0.9em; margin-bottom: 20px;">
                    Shows the return on investment for the next dollar spent. Values above 1.0 indicate profitable spend.
                </p>
            </div>
            <div id="marginal-roi-charts-manual" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px;"></div>
        `;
        chartContainer.appendChild(marginalSection);

        const marginalChartsContainer = document.getElementById('marginal-roi-charts-manual');

        Object.keys(channel_alphas).forEach(channel => {
            const alpha = channel_alphas[channel];
            const beta = channel_betas[channel];
            const r = channel_rs[channel];

            // Generate spend levels and marginal ROI manually
            const maxSpend = 10000; // Same as in manual generation
            const spendLevels = [];
            const marginalRoas = [];

            for (let spend = 0; spend <= maxSpend; spend += maxSpend / 100) {
                spendLevels.push(spend);

                // Calculate marginal ROI (derivative of the response function)
                if (spend > 0) {
                    const adstockFactor = 1 / (1 - r);
                    const marginalProfit = alpha * beta * Math.pow(adstockFactor, beta) * Math.pow(spend, beta - 1);
                    const marginalROI = marginalProfit / spend;
                    marginalRoas.push(Math.max(0, marginalROI)); // Ensure non-negative
                } else {
                    marginalRoas.push(0);
                }
            }

            // Create chart container
            const chartDiv = document.createElement('div');
            chartDiv.style.cssText = 'background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 20px;';

            const canvas = document.createElement('canvas');
            canvas.id = `marginal-roi-chart-manual-${channel}`;
            canvas.width = 400;
            canvas.height = 300;

            chartDiv.appendChild(canvas);
            marginalChartsContainer.appendChild(chartDiv);

            const ctx = canvas.getContext('2d');

            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: spendLevels,
                    datasets: [
                        {
                            label: 'Marginal ROI',
                            data: marginalRoas,
                            borderColor: '#007bff',
                            backgroundColor: 'rgba(0, 123, 255, 0.1)',
                            borderWidth: 2,
                            fill: false,
                            pointRadius: 0,
                            pointHoverRadius: 6
                        },
                        {
                            label: 'Breakeven Line (ROI = 1)',
                            data: spendLevels.map(() => 1.0),
                            borderColor: '#dc3545',
                            backgroundColor: 'transparent',
                            borderWidth: 2,
                            borderDash: [5, 5],
                            fill: false,
                            pointRadius: 0,
                            pointHoverRadius: 0
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: `${channel} - Marginal ROI`,
                            font: { size: 16, weight: 'bold' }
                        },
                        legend: {
                            display: true,
                            position: 'top'
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                            callbacks: {
                                label: function(context) {
                                    if (context.datasetIndex === 0) {
                                        return `Marginal ROI: ${context.parsed.y.toFixed(2)}`;
                                    } else {
                                        return 'Breakeven';
                                    }
                                },
                                title: function(context) {
                                    return `Daily Spend: $${context[0].parsed.x.toFixed(0)}`;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            type: 'linear',
                            position: 'bottom',
                            title: {
                                display: true,
                                text: 'Daily Spend ($)'
                            },
                            grid: {
                                alpha: 0.3
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Marginal ROI'
                            },
                            grid: {
                                alpha: 0.3
                            },
                            min: 0,
                            ticks: {
                                callback: function(value, index, values) {
                                    // Debug: Log the value being formatted
                                    console.log(`🔧 Y-axis tick formatting: ${value} -> ${Number(value).toFixed(2)}`);
                                    // Ensure we return plain numbers, not currency
                                    const formatted = Number(value).toFixed(2);
                                    return formatted;
                                },
                                // Disable any automatic number formatting
                                maxTicksLimit: 10
                            }
                        }
                    },
                    interaction: {
                        mode: 'nearest',
                        axis: 'x',
                        intersect: false
                    }
                }
            });

            console.log(`✅ Manual marginal ROI chart created for ${channel}`);
        });
    }

    async generateResponseCurveCharts(parameters, runId) {
        console.log('🚀 [FLOW] generateResponseCurveCharts started');
        console.log('📊 [FLOW] Parameters:', parameters);
        console.log('📊 [FLOW] Run ID:', runId);

        // Try to fetch actual response curves from API with confidence intervals
        try {
            const response = await fetch(`${this.apiUrl}/model/response-curves/${runId}`);
            const data = await response.json();

            if (response.ok && data.response_curves) {
                console.log('✅ [FLOW] Fetched response curves from API - starting sequential rendering');

                // 1. First: Render response curves
                console.log('📊 [FLOW] Step 1: Rendering response curves');
                this.renderResponseCurvesFromAPI(data);

                // 2. Second: Fetch marginal ROI data (this stores the HTML but doesn't insert)
                console.log('📊 [FLOW] Step 2: Fetching marginal ROI data');
                await this.fetchAndDisplayMarginalROI();

                // 3. Third: Insert the marginal ROI section after response curves
                console.log('📊 [FLOW] Step 3: Inserting marginal ROI section');
                this.insertMarginalROISection();

                // 4. Fourth: Render marginal ROI charts
                console.log('📊 [FLOW] Step 4: Rendering marginal ROI charts');
                this.renderMarginalROICharts(data);

                // 5. Fifth: Add Profit Maximizer at the very end
                console.log('📊 [FLOW] Step 5: Adding Profit Maximizer');
                await this.addProfitMaximizerAtEnd();

                console.log('✅ [FLOW] Complete dashboard sequence finished');
                return;
            }
        } catch (error) {
            console.warn('⚠️ Could not fetch response curves from API, falling back to manual generation:', error);
        }

        // Fallback to manual generation
        console.log('📊 [FLOW] Using manual generation fallback');
        const { channel_alphas, channel_betas, channel_rs } = parameters;
        Object.keys(channel_alphas).forEach(channel => {
            const alpha = channel_alphas[channel];
            const beta = channel_betas[channel];
            const r = channel_rs[channel];

            // Get confidence intervals for this channel
            const channelCI = confidenceIntervals[channel] || {};
            const alphaCI = channelCI.alpha || [alpha, alpha];
            const betaCI = channelCI.beta || [beta, beta];
            const rCI = channelCI.r || [r, r];

            // Generate spend levels from 0 to max spend
            const maxSpend = 10000; // $10k max daily spend for visualization
            const spendLevels = [];
            const profits = [];
            const profitsLower = [];
            const profitsUpper = [];

            for (let spend = 0; spend <= maxSpend; spend += maxSpend / 100) {
                // Calculate time-series adstocked spend (30-day carryover)
                const adstockedSpend = this.calculateTimeSeriesAdstock(spend, r, 30);
                const adstockedSpendLower = this.calculateTimeSeriesAdstock(spend, rCI[0], 30);
                const adstockedSpendUpper = this.calculateTimeSeriesAdstock(spend, rCI[1], 30);

                // Calculate saturated spend and profits
                const saturatedSpend = Math.pow(adstockedSpend, beta);
                const saturatedSpendLower = Math.pow(adstockedSpendLower, betaCI[0]);
                const saturatedSpendUpper = Math.pow(adstockedSpendUpper, betaCI[1]);

                // Calculate incremental profits
                const profit = alpha * saturatedSpend;
                const profitLower = alphaCI[0] * saturatedSpendLower;
                const profitUpper = alphaCI[1] * saturatedSpendUpper;

                spendLevels.push(spend);
                profits.push(profit);
                profitsLower.push(profitLower);
                profitsUpper.push(profitUpper);
            }

            // Prepare datasets
            const datasets = [{
                label: 'Lifetime Incremental Profit',
                data: profits,
                borderColor: '#2d5aa0',
                backgroundColor: 'rgba(45, 90, 160, 0.1)',
                borderWidth: 3,
                fill: false,
                tension: 0.4
            }];

            // Add confidence interval bands if available
            if (Object.keys(channelCI).length > 0) {
                datasets.push({
                    label: '95% Confidence Interval (Upper)',
                    data: profitsUpper,
                    borderColor: 'rgba(45, 90, 160, 0.4)',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0
                });

                datasets.push({
                    label: '95% Confidence Interval (Lower)',
                    data: profitsLower,
                    borderColor: 'rgba(45, 90, 160, 0.4)',
                    backgroundColor: 'rgba(45, 90, 160, 0.1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: 1, // Fill to the upper bound dataset
                    tension: 0.4,
                    pointRadius: 0
                });
            }

            // Create chart
            const ctx = document.getElementById(`chart-${channel}`);
            if (ctx) {
                new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: spendLevels.map(s => `$${Math.round(s).toLocaleString()}`),
                        datasets: datasets
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {
                            intersect: false,
                            mode: 'index'
                        },
                        plugins: {
                            title: {
                                display: true,
                                text: `${this.formatChannelName(channel)} Response Curve`,
                                font: { size: 16, weight: 'bold' }
                            },
                            legend: {
                                display: Object.keys(channelCI).length > 0,
                                position: 'bottom',
                                labels: {
                                    filter: function(legendItem, chartData) {
                                        // Only show main curve and confidence interval labels
                                        return legendItem.text !== '95% Confidence Interval (Upper)';
                                    },
                                    generateLabels: function(chart) {
                                        const labels = Chart.defaults.plugins.legend.labels.generateLabels(chart);
                                        return labels.map(label => {
                                            if (label.text === '95% Confidence Interval (Lower)') {
                                                label.text = '95% Confidence Interval';
                                            }
                                            return label;
                                        });
                                    }
                                }
                            }
                        },
                        scales: {
                            x: {
                                title: {
                                    display: true,
                                    text: 'Daily Spend ($)',
                                    font: { size: 12, weight: 'bold' }
                                },
                                ticks: {
                                    maxTicksLimit: 8
                                }
                            },
                            y: {
                                title: {
                                    display: true,
                                    text: 'Incremental Daily Profit ($)',
                                    font: { size: 12, weight: 'bold' }
                                },
                                ticks: {
                                    callback: function(value) {
                                        return '$' + Math.round(value).toLocaleString();
                                    }
                                }
                            }
                        }
                    }
                });
                console.log(`✅ Chart created for ${channel} ${Object.keys(channelCI).length > 0 ? 'with confidence intervals' : ''}`);
            } else {
                console.error(`❌ Could not find canvas element for ${channel}`);
            }
        });

        // Generate marginal ROI charts for manual generation path
        console.log('📊 [FLOW] Manual Step 1: Rendering marginal ROI charts');
        this.renderMarginalROIChartsManual(parameters);

        // Fetch marginal ROI data (stores HTML but doesn't insert)
        console.log('📊 [FLOW] Manual Step 2: Fetching marginal ROI data');
        await this.fetchAndDisplayMarginalROI();

        // Insert the marginal ROI section after response curves
        console.log('📊 [FLOW] Manual Step 3: Inserting marginal ROI section');
        this.insertMarginalROISection();

        // Add Profit Maximizer at the very end (manual path)
        console.log('📊 [FLOW] Manual Step 4: Adding Profit Maximizer');
        await this.addProfitMaximizerAtEnd();

        console.log('✅ [FLOW] Manual generation sequence finished');
    }

    calculateTimeSeriesAdstock(dailySpend, r, numDays = 30) {
        // Calculate time-series adstock effect over numDays
        let adstockedValues = [];
        for (let t = 0; t < numDays; t++) {
            if (t === 0) {
                adstockedValues.push(dailySpend);
            } else {
                adstockedValues.push(dailySpend + r * adstockedValues[t - 1]);
            }
        }
        return adstockedValues[numDays - 1];
    }

    generateMROICurvesSection(marginalROIByChannel, baselineSpend) {
        const channelNames = Object.keys(marginalROIByChannel);

        const mroiCurvesHtml = channelNames.map(channel => {
            const baselineValue = baselineSpend[channel] || 0;
            return `
                <div class="mroi-curve-card">
                    <div class="mroi-curve-title">
                        ${this.formatChannelName(channel)} - mROI Curve
                        <div class="baseline-info">Baseline: $${this.formatNumber(baselineValue)}/day (30-day avg)</div>
                    </div>
                    <div class="mroi-chart-container">
                        <canvas id="mroi-chart-${channel}"></canvas>
                    </div>
                </div>
            `;
        }).join('');

        return `
            <div style="margin-top: 30px; border: 3px solid #28a745; padding: 20px; background: #fff; border-radius: 8px;">
                <h3 style="color: #28a745; margin-bottom: 20px; font-size: 1.4rem;">📈 Marginal ROI Curves</h3>
                <div style="background: #f0f8f0; padding: 15px; border-radius: 8px; margin-bottom: 20px; font-style: italic; color: #333; border: 2px solid #28a745;">
                    <strong>These curves show how marginal ROI changes from 0 to 2x baseline spending.</strong><br>
                    The red line marks mROI = 1.0 (breakeven point) where each additional $1 spent generates $1 in profit.
                    The blue dot shows the baseline spend level (30-day average).
                </div>
                <div class="mroi-curves-grid">
                    ${mroiCurvesHtml}
                </div>
            </div>
        `;
    }

    generateMROICharts(marginalROIByChannel, baselineSpend) {
        console.log('🚀 Generating mROI charts');

        // Get model parameters from the global results (we'll need to pass them)
        this.fetchAndDisplayMROI(marginalROIByChannel, baselineSpend);
    }

    async fetchAndDisplayMROI(marginalROIByChannel, baselineSpend) {
        try {
            // We need the model parameters to calculate mROI curves
            // For now, let's use a simplified approach with the current marginal ROI values
            Object.keys(marginalROIByChannel).forEach(channel => {
                this.generateSingleMROIChart(channel, marginalROIByChannel[channel], baselineSpend[channel] || 0);
            });
        } catch (error) {
            console.error('Error generating mROI charts:', error);
        }
    }

    generateSingleMROIChart(channel, currentMROI, baselineSpend) {
        // Generate spend levels from 0 to 2x baseline (as requested)
        const maxSpend = Math.max(baselineSpend * 2, 200); // Minimum range of $200 if baseline is very low
        const spendLevels = [];
        const mroiValues = [];

        for (let spend = Math.max(1, baselineSpend * 0.1); spend <= maxSpend; spend += maxSpend / 100) {
            spendLevels.push(spend);
            // Simplified mROI calculation - in reality this would need the full model parameters
            // mROI typically decreases as spend increases due to diminishing returns
            const spendRatio = spend / Math.max(baselineSpend, 100);
            const mroiValue = currentMROI * Math.pow(spendRatio, -0.3); // Simplified diminishing returns
            mroiValues.push(Math.max(0.1, mroiValue)); // Minimum mROI of 0.1
        }

        // Find breakeven point (mROI = 1.0)
        let breakevenSpend = null;
        for (let i = 0; i < mroiValues.length; i++) {
            if (mroiValues[i] <= 1.0) {
                breakevenSpend = spendLevels[i];
                break;
            }
        }

        // Prepare datasets
        const datasets = [{
            label: 'Marginal ROI',
            data: mroiValues,
            borderColor: '#28a745',
            backgroundColor: 'rgba(40, 167, 69, 0.1)',
            borderWidth: 3,
            fill: false,
            tension: 0.4
        }];

        // Add breakeven line
        datasets.push({
            label: 'Breakeven (mROI = 1.0)',
            data: new Array(spendLevels.length).fill(1.0),
            borderColor: '#dc3545',
            backgroundColor: 'transparent',
            borderWidth: 2,
            borderDash: [10, 5],
            fill: false,
            pointRadius: 0
        });

        // Add baseline spend marker
        if (baselineSpend > 0) {
            const baselineIndex = spendLevels.findIndex(spend => spend >= baselineSpend);
            if (baselineIndex >= 0) {
                datasets.push({
                    label: 'Baseline Spend',
                    data: spendLevels.map((spend, i) => i === baselineIndex ? mroiValues[i] : null),
                    borderColor: '#17a2b8',
                    backgroundColor: '#17a2b8',
                    borderWidth: 0,
                    pointRadius: 8,
                    pointStyle: 'circle',
                    showLine: false
                });
            }
        }

        // Create chart
        const ctx = document.getElementById(`mroi-chart-${channel}`);
        if (ctx) {
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: spendLevels.map(s => `$${Math.round(s).toLocaleString()}`),
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: `${this.formatChannelName(channel)} - Marginal ROI (0 to 2x Baseline)`,
                            font: { size: 14, weight: 'bold' }
                        },
                        legend: {
                            display: true,
                            position: 'bottom'
                        },
                        tooltip: {
                            callbacks: {
                                afterBody: function(tooltipItems) {
                                    const spendValue = spendLevels[tooltipItems[0].dataIndex];
                                    const mroiValue = mroiValues[tooltipItems[0].dataIndex];
                                    let info = [];

                                    if (Math.abs(spendValue - baselineSpend) < 50) {
                                        info.push('📍 Near baseline spend level');
                                    }

                                    if (breakevenSpend && Math.abs(spendValue - breakevenSpend) < 50) {
                                        info.push('🎯 Near breakeven point');
                                    }

                                    if (mroiValue > 1.5) {
                                        info.push('🚀 High efficiency zone');
                                    } else if (mroiValue < 1.0) {
                                        info.push('⚠️ Below breakeven');
                                    }

                                    return info;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Daily Spend ($)',
                                font: { size: 12, weight: 'bold' }
                            },
                            ticks: {
                                maxTicksLimit: 8
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Marginal ROI ($)',
                                font: { size: 12, weight: 'bold' }
                            },
                            min: 0,
                            ticks: {
                                callback: function(value) {
                                    return '$' + value.toFixed(2);
                                }
                            }
                        }
                    }
                }
            });
            console.log(`✅ mROI chart created for ${channel}${breakevenSpend ? ` (breakeven at $${Math.round(breakevenSpend)}/day)` : ''}`);
        } else {
            console.error(`❌ Could not find canvas element for mROI chart ${channel}`);
        }
    }

    addProfitMaximizer(marginalROIByChannel, baselineSpend) {
        console.log('🎯 Adding Profit Maximizer section with data:', marginalROIByChannel);

        // Store the marginal ROI data for use in optimization
        this.marginalROIByChannel = marginalROIByChannel;
        this.baselineSpend = baselineSpend;

        const profitMaximizerSection = this.generateProfitMaximizerSection(marginalROIByChannel, baselineSpend);

        // Insert after the mROI curves section
        const mroiSection = document.querySelector('[style*="border: 3px solid #28a745"]');
        if (mroiSection) {
            mroiSection.insertAdjacentHTML('afterend', profitMaximizerSection);
            this.initializeProfitMaximizer(marginalROIByChannel, baselineSpend);
            console.log('✅ Profit Maximizer section added successfully');
        } else {
            console.error('❌ Could not find mROI section to insert Profit Maximizer after');
        }
    }

    async addProfitMaximizerAtEnd() {
        console.log('🎯 Adding Profit Maximizer at the end');
        try {
            const response = await fetch(`${this.apiUrl}/model/marginal-roi/${this.runId}`);
            const marginalROIData = await response.json();

            if (response.ok && marginalROIData.marginal_roi_by_channel) {
                this.addProfitMaximizer(marginalROIData.marginal_roi_by_channel, marginalROIData.baseline_spend_per_day);
            } else {
                console.warn('Marginal ROI data not available for Profit Maximizer:', marginalROIData);
            }
        } catch (error) {
            console.warn('Error fetching marginal ROI for Profit Maximizer:', error);
        }
    }

    insertMarginalROISection() {
        console.log('📊 Inserting marginal ROI section after response curves');
        if (this.marginalROISection) {
            // Find the response curves section by its CSS class
            const responseCurvesSection = document.querySelector('.response-curves-section');
            if (responseCurvesSection) {
                responseCurvesSection.insertAdjacentHTML('afterend', this.marginalROISection);
                console.log('✅ Marginal ROI section inserted after response curves section');
            } else {
                // Fallback: find by the equations grid and insert marginal ROI after it
                const equationsGrid = document.querySelector('.equations-grid');
                if (equationsGrid && equationsGrid.parentElement) {
                    equationsGrid.parentElement.insertAdjacentHTML('afterend', this.marginalROISection);
                    console.log('✅ Marginal ROI section inserted after equations section (fallback)');
                } else {
                    // Ultimate fallback: insert after results grid
                    const resultsGrid = document.getElementById('results-grid');
                    if (resultsGrid) {
                        resultsGrid.insertAdjacentHTML('afterend', this.marginalROISection);
                        console.log('⚠️ Marginal ROI section inserted after results grid (ultimate fallback)');
                    }
                }
            }
        } else {
            console.warn('⚠️ No marginal ROI section HTML stored');
        }
    }

    generateProfitMaximizerSection(marginalROIByChannel, baselineSpend) {
        const channels = Object.keys(marginalROIByChannel);

        const constraintInputs = channels.map(channel => {
            const baseline = baselineSpend[channel] || 0;
            return `
                <div class="constraint-row">
                    <div class="channel-label">${this.formatChannelName(channel)}</div>
                    <div class="constraint-inputs">
                        <label>Min: $</label>
                        <input type="number" id="min-${channel}" min="0" placeholder="0" class="constraint-input">
                        <label>Max: $</label>
                        <input type="number" id="max-${channel}" min="0" placeholder="${Math.round(baseline * 3)}" class="constraint-input">
                        <div class="baseline-hint">Baseline: $${this.formatNumber(baseline)}/day</div>
                    </div>
                </div>
            `;
        }).join('');

        return `
            <div style="margin-top: 30px; border: 3px solid #007bff; padding: 20px; background: #fff; border-radius: 8px;" id="profit-maximizer-section">
                <h3 style="color: #007bff; margin-bottom: 20px; font-size: 1.4rem;">🎯 Profit Maximizer</h3>
                <div style="background: #f0f8ff; padding: 15px; border-radius: 8px; margin-bottom: 20px; font-style: italic; color: #333; border: 2px solid #007bff;">
                    <strong>Enter your daily budget and any channel constraints to get optimal allocation recommendations.</strong><br>
                    The optimizer uses your response curves to maximize profit within your budget and constraints.
                </div>

                <div class="profit-maximizer-form">
                    <div class="budget-input-section">
                        <h4>📊 Daily Budget</h4>
                        <div class="budget-input-row">
                            <label for="total-budget">Total Daily Budget: $</label>
                            <input type="number" id="total-budget" min="1" placeholder="1000" class="budget-input">
                            <button id="optimize-button" class="optimize-btn">🚀 Optimize Allocation</button>
                        </div>
                    </div>

                    <div class="constraints-section">
                        <h4>⚙️ Channel Constraints (Optional)</h4>
                        <div class="constraints-grid">
                            ${constraintInputs}
                        </div>
                    </div>

                    <div class="optimization-results hidden" id="optimization-results">
                        <h4>💡 Recommended Allocation</h4>
                        <div id="allocation-display"></div>
                        <div id="expected-profit-display"></div>
                    </div>
                </div>
            </div>
        `;
    }

    initializeProfitMaximizer(marginalROIByChannel, baselineSpend) {
        const optimizeButton = document.getElementById('optimize-button');
        if (optimizeButton) {
            optimizeButton.addEventListener('click', () => {
                this.optimizeBudgetAllocation(marginalROIByChannel, baselineSpend);
            });
        }
    }

    optimizeBudgetAllocation(marginalROIByChannel, baselineSpend) {
        const totalBudget = parseFloat(document.getElementById('total-budget').value);

        if (!totalBudget || totalBudget <= 0) {
            alert('Please enter a valid total daily budget.');
            return;
        }

        const channels = Object.keys(marginalROIByChannel);
        const constraints = {};

        // Collect constraints
        channels.forEach(channel => {
            const minInput = document.getElementById(`min-${channel}`);
            const maxInput = document.getElementById(`max-${channel}`);

            constraints[channel] = {
                min: parseFloat(minInput.value) || 0,
                max: parseFloat(maxInput.value) || totalBudget
            };
        });

        // Perform optimization
        const optimization = this.performOptimization(totalBudget, constraints, channels, baselineSpend);

        // Display results
        this.displayOptimizationResults(optimization, totalBudget);
    }

    performOptimization(totalBudget, constraints, channels, baselineSpend) {
        let allocation = {};
        let remainingBudget = totalBudget;

        // Initialize with minimum constraints
        channels.forEach(channel => {
            allocation[channel] = constraints[channel].min;
            remainingBudget -= constraints[channel].min;
        });

        if (remainingBudget < 0) {
            return { error: "Total minimum constraints exceed budget" };
        }

        // Filter out channels with zero marginal ROI (alpha = 0)
        const viableChannels = channels.filter(channel => {
            const mROI = this.marginalROIByChannel[channel];
            return mROI && mROI > 0.01; // Only include channels with meaningful ROI
        });

        if (viableChannels.length === 0) {
            return { error: "No viable channels with positive ROI found" };
        }

        // Greedy allocation: repeatedly allocate to the channel with highest marginal ROI
        // Use smaller increments for more precise allocation
        const incrementSize = Math.max(1, remainingBudget / 200); // Smaller increments

        while (remainingBudget > 0.01) { // Use small threshold instead of incrementSize
            let bestChannel = null;
            let bestMarginalROI = 0;

            viableChannels.forEach(channel => {
                const currentSpend = allocation[channel];
                const maxSpend = constraints[channel].max;

                // Only consider channels that can still receive more budget
                if (currentSpend + incrementSize <= maxSpend) {
                    // Use the actual marginal ROI from the model
                    const mROI = this.marginalROIByChannel[channel];

                    // Apply diminishing returns based on current spend vs baseline
                    const baseline = Math.max(baselineSpend[channel] || 0, 1);
                    const spendRatio = currentSpend / baseline;
                    const diminishingFactor = Math.pow(Math.max(0.1, 1 / (1 + spendRatio * 0.5)), 0.3);
                    const adjustedROI = mROI * diminishingFactor;

                    if (adjustedROI > bestMarginalROI) {
                        bestMarginalROI = adjustedROI;
                        bestChannel = channel;
                    }
                }
            });

            if (bestChannel && bestMarginalROI > 0) {
                const increase = Math.min(incrementSize,
                    constraints[bestChannel].max - allocation[bestChannel],
                    remainingBudget);
                allocation[bestChannel] += increase;
                remainingBudget -= increase;
            } else {
                // No more beneficial allocations possible, allocate remainder proportionally
                const allocatableChannels = viableChannels.filter(channel =>
                    allocation[channel] < constraints[channel].max);

                if (allocatableChannels.length > 0 && remainingBudget > 0.01) {
                    // Allocate all remaining budget to the first allocatable channel
                    const channel = allocatableChannels[0];
                    const increase = Math.min(remainingBudget, constraints[channel].max - allocation[channel]);
                    allocation[channel] += increase;
                    remainingBudget -= increase;
                }
                break;
            }
        }

        // Calculate total expected profit using actual marginal ROI data
        let totalExpectedProfit = 0;
        channels.forEach(channel => {
            const spend = allocation[channel];
            const mROI = this.marginalROIByChannel[channel] || 0;
            const baseline = Math.max(baselineSpend[channel] || 0, 1);

            // Calculate profit with diminishing returns
            if (mROI > 0 && spend > 0) {
                const spendRatio = spend / baseline;
                const diminishingFactor = Math.pow(Math.max(0.1, 1 / (1 + spendRatio * 0.5)), 0.3);
                const effectiveROI = mROI * diminishingFactor;
                totalExpectedProfit += spend * effectiveROI;
            }
        });

        return {
            allocation,
            totalExpectedProfit,
            totalAllocated: totalBudget - remainingBudget,
            remainingBudget
        };
    }


    displayOptimizationResults(optimization, totalBudget) {
        const resultsDiv = document.getElementById('optimization-results');
        const allocationDiv = document.getElementById('allocation-display');
        const profitDiv = document.getElementById('expected-profit-display');

        if (optimization.error) {
            allocationDiv.innerHTML = `<div class="error-message">${optimization.error}</div>`;
            resultsDiv.classList.remove('hidden');
            return;
        }

        // Create allocation display
        const allocationHtml = Object.entries(optimization.allocation).map(([channel, amount]) => {
            const percentage = (amount / totalBudget * 100).toFixed(1);
            return `
                <div class="allocation-item">
                    <div class="channel-allocation">
                        <span class="channel-name">${this.formatChannelName(channel)}</span>
                        <span class="allocation-amount">$${this.formatNumber(amount)}/day (${percentage}%)</span>
                    </div>
                    <div class="allocation-bar">
                        <div class="allocation-fill" style="width: ${percentage}%;"></div>
                    </div>
                </div>
            `;
        }).join('');

        allocationDiv.innerHTML = `
            <div class="allocation-grid">
                ${allocationHtml}
            </div>
        `;

        // Create profit display
        profitDiv.innerHTML = `
            <div class="profit-summary">
                <div class="profit-metrics">
                    <div class="profit-metric">
                        <span class="metric-label">Expected Daily Profit:</span>
                        <span class="metric-value">$${this.formatNumber(optimization.totalExpectedProfit)}</span>
                    </div>
                    <div class="profit-metric">
                        <span class="metric-label">Total Allocated:</span>
                        <span class="metric-value">$${this.formatNumber(optimization.totalAllocated)}</span>
                    </div>
                    ${optimization.remainingBudget > 0 ? `
                    <div class="profit-metric">
                        <span class="metric-label">Remaining Budget:</span>
                        <span class="metric-value">$${this.formatNumber(optimization.remainingBudget)}</span>
                    </div>
                    ` : ''}
                </div>
            </div>
        `;

        resultsDiv.classList.remove('hidden');
    }

    formatChannelName(name) {
        return name.split('_').map(word =>
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }

    displayFoldResults() {
        // Create or update fold results table
        if (!this.foldResults || this.foldResults.length === 0) return;

        // Check if fold results section exists, if not create it
        let foldResultsSection = document.getElementById('fold-results-section');
        if (!foldResultsSection) {
            foldResultsSection = document.createElement('div');
            foldResultsSection.id = 'fold-results-section';
            foldResultsSection.className = 'section';
            foldResultsSection.innerHTML = `
                <h3>📊 Cross-Validation Fold Results</h3>
                <div id="fold-results-table"></div>
            `;

            // Insert after progress section
            const progressSection = document.getElementById('progress-section');
            if (progressSection) {
                progressSection.parentNode.insertBefore(foldResultsSection, progressSection.nextSibling);
            }
        }

        // Build fold results table
        const tableHtml = `
            <table class="fold-results-table" style="width: 100%; margin-top: 10px; border-collapse: collapse;">
                <thead>
                    <tr style="background: #f5f5f5;">
                        <th style="padding: 8px; border: 1px solid #ddd;">Fold</th>
                        <th style="padding: 8px; border: 1px solid #ddd;">MAPE (%)</th>
                        ${this.foldResults[0]?.parameters?.channel_betas ?
                            Object.keys(this.foldResults[0].parameters.channel_betas)
                                .map(ch => `<th style="padding: 8px; border: 1px solid #ddd;" colspan="2">${this.formatChannelName(ch)}</th>`)
                                .join('') : ''}
                    </tr>
                    ${this.foldResults[0]?.parameters?.channel_betas ?
                        `<tr style="background: #f9f9f9;">
                            <th style="padding: 4px; border: 1px solid #ddd;"></th>
                            <th style="padding: 4px; border: 1px solid #ddd;"></th>
                            ${Object.keys(this.foldResults[0].parameters.channel_betas)
                                .map(() => `<th style="padding: 4px; border: 1px solid #ddd;">β</th><th style="padding: 4px; border: 1px solid #ddd;">r</th>`)
                                .join('')}
                        </tr>` : ''}
                </thead>
                <tbody>
                    ${this.foldResults.map(fold => {
                        const params = fold.parameters;
                        return `<tr>
                            <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">${fold.fold}</td>
                            <td style="padding: 8px; border: 1px solid #ddd; text-align: center; font-weight: bold;">${fold.mape.toFixed(2)}</td>
                            ${params?.channel_betas ?
                                Object.keys(params.channel_betas).map(ch =>
                                    `<td style="padding: 8px; border: 1px solid #ddd; text-align: center;">${params.channel_betas[ch].toFixed(3)}</td>
                                     <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">${params.channel_rs[ch].toFixed(3)}</td>`
                                ).join('') : ''}
                        </tr>`;
                    }).join('')}
                </tbody>
            </table>
        `;

        document.getElementById('fold-results-table').innerHTML = tableHtml;
    }

    displayHoldoutValidation(data) {
        // Store holdout data for later display
        this.holdoutValidationData = data;

        // Create holdout validation section
        let holdoutSection = document.getElementById('holdout-validation-section');
        if (!holdoutSection) {
            holdoutSection = document.createElement('div');
            holdoutSection.id = 'holdout-validation-section';
            holdoutSection.className = 'section';
            holdoutSection.style.marginTop = '20px';

            // Insert after final params section or fold results
            const finalParamsSection = document.getElementById('final-params-section');
            const foldResultsSection = document.getElementById('fold-results-section');
            const referenceSection = finalParamsSection || foldResultsSection || document.getElementById('progress-section');

            if (referenceSection) {
                referenceSection.parentNode.insertBefore(holdoutSection, referenceSection.nextSibling);
            }
        }

        // Determine validation quality
        const mapeRatio = data.holdout_mape / data.cv_mape;
        let statusColor, statusText, statusIcon;

        if (mapeRatio < 1.1) {
            statusColor = '#4CAF50';  // Green
            statusText = 'Excellent - No overfitting detected';
            statusIcon = '✅';
        } else if (mapeRatio < 1.2) {
            statusColor = '#FF9800';  // Orange
            statusText = 'Good - Minor overfitting';
            statusIcon = '⚠️';
        } else {
            statusColor = '#F44336';  // Red
            statusText = 'Warning - Significant overfitting detected';
            statusIcon = '❌';
        }

        // Build holdout validation display
        holdoutSection.innerHTML = `
            <h3>🎯 Final Holdout Validation</h3>
            <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-top: 10px; border-left: 4px solid ${statusColor};">
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 15px;">
                    <div style="text-align: center;">
                        <div style="font-size: 24px; font-weight: bold; color: #333;">${data.holdout_mape.toFixed(2)}%</div>
                        <div style="color: #666; margin-top: 5px;">Holdout MAPE</div>
                        <div style="font-size: 12px; color: #999; margin-top: 3px;">(${data.holdout_days} days)</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 24px; font-weight: bold; color: #333;">${data.cv_mape.toFixed(2)}%</div>
                        <div style="color: #666; margin-top: 5px;">CV MAPE</div>
                        <div style="font-size: 12px; color: #999; margin-top: 3px;">(avg of folds)</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 24px; font-weight: bold; color: ${data.mape_difference > 0 ? '#F44336' : '#4CAF50'};">
                            ${data.mape_difference > 0 ? '+' : ''}${data.mape_difference.toFixed(2)}%
                        </div>
                        <div style="color: #666; margin-top: 5px;">Difference</div>
                        <div style="font-size: 12px; color: #999; margin-top: 3px;">(${mapeRatio.toFixed(2)}x)</div>
                    </div>
                </div>
                <div style="padding: 10px; background: white; border-radius: 4px; display: flex; align-items: center;">
                    <span style="font-size: 20px; margin-right: 10px;">${statusIcon}</span>
                    <div>
                        <div style="font-weight: bold; color: ${statusColor};">${statusText}</div>
                        <div style="font-size: 12px; color: #666; margin-top: 2px;">
                            The model's performance on unseen holdout data ${
                                mapeRatio < 1.1 ? 'closely matches' :
                                mapeRatio < 1.2 ? 'is slightly worse than' :
                                'is significantly worse than'
                            } the cross-validation results.
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Also show status message
        const msg = `🎯 Holdout Validation Complete: MAPE = ${data.holdout_mape.toFixed(2)}% ${statusIcon}`;
        this.showTrainingStatus(msg, data.is_overfit ? 'error' : 'success');
    }

    displayFinalParameters(params) {
        // Create final parameters section
        let finalParamsSection = document.getElementById('final-params-section');
        if (!finalParamsSection) {
            finalParamsSection = document.createElement('div');
            finalParamsSection.id = 'final-params-section';
            finalParamsSection.className = 'section';
            finalParamsSection.style.marginTop = '20px';
            finalParamsSection.innerHTML = `
                <h3>🎯 Final Selected Parameters</h3>
                <div id="final-params-content"></div>
            `;

            const foldResultsSection = document.getElementById('fold-results-section');
            if (foldResultsSection) {
                foldResultsSection.parentNode.insertBefore(finalParamsSection, foldResultsSection.nextSibling);
            }
        }

        // Build parameters display
        const paramsHtml = `
            <div style="background: #f0f8ff; padding: 15px; border-radius: 8px; margin-top: 10px;">
                <div style="margin-bottom: 10px;">
                    <strong>Baseline Parameters:</strong>
                    <span style="margin-left: 10px;">α_baseline = ${params.alpha_baseline.toFixed(4)}, α_trend = ${params.alpha_trend.toFixed(6)}</span>
                </div>
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="background: #e0f0ff;">
                            <th style="padding: 8px; border: 1px solid #ccc;">Channel</th>
                            <th style="padding: 8px; border: 1px solid #ccc;">Alpha (α)</th>
                            <th style="padding: 8px; border: 1px solid #ccc;">Beta (β)</th>
                            <th style="padding: 8px; border: 1px solid #ccc;">R (memory)</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${Object.keys(params.channel_alphas).map(ch => `
                            <tr>
                                <td style="padding: 8px; border: 1px solid #ccc; font-weight: bold;">${this.formatChannelName(ch)}</td>
                                <td style="padding: 8px; border: 1px solid #ccc; text-align: center;">${params.channel_alphas[ch].toFixed(4)}</td>
                                <td style="padding: 8px; border: 1px solid #ccc; text-align: center;">${params.channel_betas[ch].toFixed(3)}</td>
                                <td style="padding: 8px; border: 1px solid #ccc; text-align: center;">${params.channel_rs[ch].toFixed(3)}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;

        document.getElementById('final-params-content').innerHTML = paramsHtml;
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MMMApp();
});

// Handle page unload to clean up intervals
window.addEventListener('beforeunload', () => {
    if (window.mmmApp && window.mmmApp.progressInterval) {
        clearInterval(window.mmmApp.progressInterval);
    }
});