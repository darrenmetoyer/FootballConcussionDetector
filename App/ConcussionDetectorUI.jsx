const { useEffect, useRef, useState } = React;

const googleClientId = window.GOOGLE_CLIENT_ID || "";

function parseJwt(token) {
  try {
    const base64Url = token.split(".")[1];
    const base64 = base64Url.replace(/-/g, "+").replace(/_/g, "/");
    const jsonPayload = decodeURIComponent(
      atob(base64)
        .split("")
        .map((char) => "%" + ("00" + char.charCodeAt(0).toString(16)).slice(-2))
        .join("")
    );
    return JSON.parse(jsonPayload);
  } catch (error) {
    console.error("[parseJwt] failed:", error);
    return null;
  }
}

function loadStoredJson(key, fallbackValue) {
  try {
    const storedValue = localStorage.getItem(key);
    return storedValue ? JSON.parse(storedValue) : fallbackValue;
  } catch (error) {
    console.error(`[loadStoredJson] failed for ${key}:`, error);
    return fallbackValue;
  }
}

function Banner({ type, message }) {
  if (!message) return null;
  return <div className={`banner ${type}`}>{message}</div>;
}

function App() {
  const [currentPage, setCurrentPage] = useState(() => localStorage.getItem("cdPage") || "home");
  const [user, setUser] = useState(() => loadStoredJson("cdUser", null));
  const [menuOpen, setMenuOpen] = useState(false);
  const [reports, setReports] = useState([]);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [authError, setAuthError] = useState("");
  const [appError, setAppError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [progressPercent, setProgressPercent] = useState(0);
  const [progressMessage, setProgressMessage] = useState("Waiting to start...");

  const googleButtonRef = useRef(null);
  const fileInputRef = useRef(null);
  const progressIntervalRef = useRef(null);

  useEffect(() => {
    localStorage.setItem("cdPage", currentPage);
  }, [currentPage]);

  useEffect(() => {
    try {
      if (user) {
        localStorage.setItem("cdUser", JSON.stringify(user));
      } else {
        localStorage.removeItem("cdUser");
      }
    } catch (error) {
      console.warn("[persistUser] failed:", error);
    }
  }, [user]);

  useEffect(() => {
    console.log("[app] user changed:", user);
    if (!user?.id) return;
    fetchReports();
  }, [user]);

  useEffect(() => {
    if (!googleClientId) {
      setAuthError("Google Sign-In is not configured on the server.");
      console.error("[google] missing GOOGLE_CLIENT_ID");
      return;
    }

    let isCancelled = false;

    const initializeGoogle = () => {
      if (isCancelled) return;

      if (!window.google?.accounts?.id) {
        console.error("[google] GIS failed to load");
        setAuthError("Google Identity Services failed to load.");
        return;
      }

      try {
        console.log("[google] initializing");
        window.google.accounts.id.initialize({
          client_id: googleClientId,
          callback: handleCredentialResponse,
          auto_select: false,
          cancel_on_tap_outside: true,
        });

        if (googleButtonRef.current) {
          googleButtonRef.current.innerHTML = "";
          window.google.accounts.id.renderButton(googleButtonRef.current, {
            type: "standard",
            theme: "outline",
            size: "large",
            text: "continue_with",
            shape: "rectangular",
            width: 320,
          });
        }

        console.log("[google] ready");
        setAuthError("");
      } catch (error) {
        console.error("[google] init error:", error);
        setAuthError("Failed to initialize Google Sign-In.");
      }
    };

    if (window.google?.accounts?.id) {
      initializeGoogle();
      return () => {
        isCancelled = true;
      };
    }

    const script = document.createElement("script");
    script.src = "https://accounts.google.com/gsi/client";
    script.async = true;
    script.defer = true;
    script.onload = initializeGoogle;
    script.onerror = () => {
      if (!isCancelled) {
        console.error("[google] script load failed");
        setAuthError("Failed to load Google Sign-In script.");
      }
    };
    document.body.appendChild(script);

    return () => {
      isCancelled = true;
    };
  }, []);

  function startFakeProgress() {
    console.log("[progress] starting");
    setProgressPercent(5);
    setProgressMessage("Uploading video...");

    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
    }

    progressIntervalRef.current = setInterval(() => {
      setProgressPercent((previousValue) => {
        if (previousValue < 20) {
          setProgressMessage("Uploading video...");
          return previousValue + 5;
        }
        if (previousValue < 55) {
          setProgressMessage("Running detection pipeline...");
          return previousValue + 3;
        }
        if (previousValue < 85) {
          setProgressMessage("Generating AI report...");
          return previousValue + 2;
        }
        if (previousValue < 95) {
          setProgressMessage("Saving report...");
          return previousValue + 1;
        }
        return previousValue;
      });
    }, 600);
  }

  function stopFakeProgress(success = true) {
    console.log("[progress] stopping");
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }

    if (success) {
      setProgressPercent(100);
      setProgressMessage("Done.");
    } else {
      setProgressMessage("Stopped.");
    }

    setTimeout(() => {
      setProgressPercent(0);
      setProgressMessage("Waiting to start...");
    }, 800);
  }

  async function fetchReports() {
    if (!user?.id) return;

    try {
      console.log("[fetchReports] starting for userId:", user.id);
      const response = await fetch(`/api/reports?userId=${encodeURIComponent(user.id)}`);
      console.log("[fetchReports] status:", response.status);

      if (!response.ok) {
        throw new Error(`Failed to fetch reports (${response.status})`);
      }

      const data = await response.json();
      console.log("[fetchReports] data:", data);
      setReports(Array.isArray(data) ? data : []);
    } catch (error) {
      console.error("[fetchReports] failed:", error);
      setReports([]);
      setAppError(error.message || "Failed to fetch reports.");
    }
  }

  async function handleCredentialResponse(response) {
    setAuthError("");
    setAppError("");
    setSuccessMessage("");

    try {
      console.log("[auth] credential response received");
      if (!response?.credential) {
        throw new Error("No Google credential returned.");
      }

      const jwtUser = parseJwt(response.credential);
      console.log("[auth] parsed jwt user:", jwtUser);

      const authResponse = await fetch("/api/auth/google", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          token: response.credential,
        }),
      });

      console.log("[auth] /api/auth/google status:", authResponse.status);
      const data = await authResponse.json();
      console.log("[auth] /api/auth/google data:", data);

      if (!authResponse.ok || !data.success) {
        throw new Error(data.error || "Google authentication failed.");
      }

      const serverUser = data.user || {};
      const nextUser = {
        id: serverUser.id,
        googleId: serverUser.googleId,
        name: serverUser.name || jwtUser?.name || "User",
        email: serverUser.email || jwtUser?.email || "",
        avatar: serverUser.picture || jwtUser?.picture || "",
      };

      console.log("[auth] login success, setting user:", nextUser);
      setUser(nextUser);
      setCurrentPage("home");
      setSuccessMessage(`Signed in as ${nextUser.name}.`);
    } catch (error) {
      console.error("[auth] sign-in failed:", error);
      setAuthError(error.message || "Sign-in failed.");
    }
  }

  function handleSignOut() {
    try {
      console.log("[auth] signing out");
      if (window.google?.accounts?.id) {
        window.google.accounts.id.disableAutoSelect();
      }
    } catch (error) {
      console.warn("[auth] sign-out cleanup failed:", error);
    }

    setUser(null);
    setReports([]);
    setUploadedFile(null);
    setCurrentPage("home");
    setSuccessMessage("Signed out.");
    setAppError("");
    setMenuOpen(false);
  }

  function navigateTo(page) {
    console.log("[nav] navigating to:", page);
    setCurrentPage(page);
    setMenuOpen(false);
    setAppError("");
    setSuccessMessage("");
  }

  function isAllowedVideoFile(file) {
    if (!file) return false;
    const fileName = file.name.toLowerCase();
    return [".mp4", ".avi", ".mov", ".mkv"].some((extension) => fileName.endsWith(extension));
  }

  function handleDragOver(event) {
    event.preventDefault();
    setIsDragging(true);
  }

  function handleDragLeave() {
    setIsDragging(false);
  }

  function handleDrop(event) {
    event.preventDefault();
    setIsDragging(false);
    setAppError("");
    setSuccessMessage("");

    const file = event.dataTransfer.files[0];
    console.log("[upload] dropped file:", file?.name);

    if (!file) return;

    if (!isAllowedVideoFile(file)) {
      setAppError("Please upload an MP4, AVI, MOV, or MKV file.");
      return;
    }

    setUploadedFile(file);
    analyzeVideo(file);
  }

  function handleFileSelect(event) {
    setAppError("");
    setSuccessMessage("");

    const file = event.target.files[0];
    console.log("[upload] selected file:", file?.name);

    if (!file) return;

    if (!isAllowedVideoFile(file)) {
      setAppError("Please upload an MP4, AVI, MOV, or MKV file.");
      return;
    }

    setUploadedFile(file);
    analyzeVideo(file);
  }

  async function analyzeVideo(file) {
    try {
      console.log("[analyzeVideo] starting for file:", file?.name);

      if (!user?.id) {
        throw new Error("You must be signed in.");
      }

      setIsAnalyzing(true);
      setAppError("");
      setSuccessMessage("");
      startFakeProgress();

      const formData = new FormData();
      formData.append("video", file);
      formData.append("userId", String(user.id));

      console.log("[analyzeVideo] posting to /api/analyze");
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      console.log("[analyzeVideo] response status:", response.status);
      const data = await response.json();
      console.log("[analyzeVideo] response data:", data);

      if (!response.ok || !data.success) {
        throw new Error(data.error || "Analysis failed.");
      }

      setSuccessMessage("Video analyzed successfully.");
      await fetchReports();
      setCurrentPage("reports");

      if (data.report?.downloadUrl) {
        console.log("[analyzeVideo] opening download:", data.report.downloadUrl);
        window.open(data.report.downloadUrl, "_blank");
      }

      stopFakeProgress(true);
    } catch (error) {
      console.error("[analyzeVideo] failed:", error);
      setAppError(error.message || "Video analysis failed.");
      stopFakeProgress(false);
    } finally {
      setIsAnalyzing(false);
    }
  }

  function LoginPage() {
    return (
      <div className="loginPage">
        <div className="loginCard">
          <div className="loginHeader">
            <h1>Concussion Detector</h1>
            <p>Sign in to access the analysis platform</p>
          </div>

          <div className="googleButtonWrap">
            <div ref={googleButtonRef} className="googleRenderedButton" />
          </div>

          <Banner type="error" message={authError} />

          {!googleClientId && (
            <div className="configWarning">
              Server did not provide <code>window.GOOGLE_CLIENT_ID</code>.
            </div>
          )}

          <div className="loginFooter">
            <p>By signing in, you agree to our Terms of Service and Privacy Policy</p>
          </div>
        </div>
      </div>
    );
  }

  function NavigationMenu() {
    return (
      <div className="navMenu">
        <button
          className="menuToggle"
          onClick={() => setMenuOpen((previousValue) => !previousValue)}
          aria-label="Toggle menu"
          type="button"
        >
          ☰
        </button>

        {menuOpen && (
          <>
            <div className="menuBackdrop" onClick={() => setMenuOpen(false)} />
            <div className="menuDropdown">
              <button onClick={() => navigateTo("home")} className="menuItem" type="button">
                Upload
              </button>
              <button onClick={() => navigateTo("reports")} className="menuItem" type="button">
                Reports
              </button>
              <button onClick={handleSignOut} className="menuItem signOut" type="button">
                Sign Out
              </button>
            </div>
          </>
        )}
      </div>
    );
  }

  function HomePage() {
    return (
      <div className="homePage">
        <div className="pageHeader">
          <h1>Upload Video</h1>
          <p>Signed in{user?.name ? ` as ${user.name}` : ""}. Upload gameplay footage to run analysis.</p>
        </div>

        <Banner type="success" message={successMessage} />
        <Banner type="error" message={appError} />

        <div
          className={`uploadZone ${isDragging ? "dragging" : ""} ${uploadedFile ? "uploaded" : ""}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".mp4,.avi,.mov,.mkv,video/*"
            onChange={handleFileSelect}
            style={{ display: "none" }}
          />

          {!uploadedFile ? (
            <>
              <div className="bigSymbol">🎥</div>
              <h3>Upload Video for Analysis</h3>
              <p>Drag and drop your video file here or click to browse</p>
              <div className="supportedFormats">Supports MP4, AVI, MOV, MKV</div>
            </>
          ) : (
            <>
              <div className="bigSymbol">✅</div>
              <h3>{uploadedFile.name}</h3>
              <p>Ready for analysis</p>
            </>
          )}
        </div>

        {isAnalyzing && (
          <div className="analyzingOverlay">
            <div className="analyzingContent">
              <div className="spinner"></div>
              <h2>Analyzing Video</h2>
              <p>{progressMessage}</p>
              <div className="progressBar">
                <div className="progressFill" style={{ width: `${progressPercent}%` }}></div>
              </div>
              <div className="progressText">{progressPercent}%</div>
            </div>
          </div>
        )}
      </div>
    );
  }

  function ReportsPage() {
    return (
      <div className="reportsPage">
        <div className="pageHeader">
          <h1>Reports</h1>
          <p>Your saved reports.</p>
        </div>

        <Banner type="success" message={successMessage} />
        <Banner type="error" message={appError} />

        <div className="reportsList">
          {reports.length === 0 ? (
            <div className="emptyState">
              <p>No saved reports yet.</p>
            </div>
          ) : (
            reports.map((report) => (
              <div key={report.id} className="reportCard">
                <div className="reportRow">
                  <div className="reportInfo">
                    <h3>{report.videoName || "Untitled Report"}</h3>
                    <div className="reportDate">
                      {report.date
                        ? new Date(report.date).toLocaleDateString("en-US", {
                            month: "long",
                            day: "numeric",
                            year: "numeric",
                          })
                        : "Unknown date"}
                    </div>
                  </div>

                  <a
                    href={report.downloadUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="downloadButton"
                  >
                    Download
                  </a>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="appContainer">
      <style>{`
        * { margin: 0; padding: 0; box-sizing: border-box; }

        :root {
          --primaryBlue: #0047ab;
          --accentCyan: #00d4ff;
          --darkNavy: #001f3f;
          --bgPrimary: #f8f9fa;
          --bgSecondary: #ffffff;
          --textPrimary: #1a2332;
          --textSecondary: #5a6c7d;
          --borderColor: #e1e8ed;
          --success: #00c896;
          --danger: #ff4757;
          --shadowMd: 0 4px 12px rgba(0, 0, 0, 0.1);
          --shadowLg: 0 8px 24px rgba(0, 0, 0, 0.12);
        }

        body {
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
          background: var(--bgPrimary);
          color: var(--textPrimary);
        }

        .appContainer {
          min-height: 100vh;
          position: relative;
        }

        .navMenu {
          position: fixed;
          top: 24px;
          right: 24px;
          z-index: 1000;
        }

        .menuToggle {
          width: 48px;
          height: 48px;
          border: none;
          border-radius: 12px;
          background: var(--bgSecondary);
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          box-shadow: var(--shadowMd);
          font-size: 24px;
        }

        .menuBackdrop {
          position: fixed;
          inset: 0;
          background: rgba(0, 0, 0, 0.2);
        }

        .menuDropdown {
          position: absolute;
          top: 56px;
          right: 0;
          width: 240px;
          background: white;
          border-radius: 12px;
          box-shadow: var(--shadowLg);
          overflow: hidden;
        }

        .menuItem {
          width: 100%;
          padding: 14px 20px;
          border: none;
          background: transparent;
          cursor: pointer;
          text-align: left;
          font-size: 15px;
        }

        .menuItem:hover {
          background: #f4f7fa;
          color: var(--primaryBlue);
        }

        .menuItem.signOut {
          border-top: 1px solid var(--borderColor);
          color: var(--danger);
        }

        .loginPage {
          min-height: 100vh;
          display: flex;
          align-items: center;
          justify-content: center;
          background: linear-gradient(135deg, var(--darkNavy) 0%, var(--primaryBlue) 100%);
          padding: 24px;
        }

        .loginCard {
          background: white;
          border-radius: 20px;
          padding: 48px;
          max-width: 440px;
          width: 100%;
          box-shadow: var(--shadowLg);
        }

        .loginHeader {
          text-align: center;
          margin-bottom: 28px;
        }

        .loginHeader h1 {
          font-size: 32px;
          margin-bottom: 8px;
        }

        .loginHeader p {
          color: var(--textSecondary);
        }

        .googleButtonWrap {
          display: flex;
          justify-content: center;
          margin-bottom: 16px;
          min-height: 44px;
        }

        .loginFooter {
          margin-top: 24px;
          text-align: center;
        }

        .loginFooter p {
          font-size: 13px;
          color: var(--textSecondary);
          line-height: 1.5;
        }

        .homePage, .reportsPage {
          max-width: 1100px;
          margin: 0 auto;
          padding: 80px 24px 60px;
        }

        .pageHeader {
          margin-bottom: 24px;
        }

        .pageHeader h1 {
          font-size: 40px;
          margin-bottom: 8px;
        }

        .pageHeader p {
          color: var(--textSecondary);
          font-size: 18px;
        }

        .banner {
          border-radius: 10px;
          padding: 12px 14px;
          font-size: 14px;
          margin-bottom: 16px;
        }

        .banner.success {
          background: rgba(0, 200, 150, 0.08);
          color: var(--success);
          border: 1px solid rgba(0, 200, 150, 0.2);
        }

        .banner.error, .configWarning {
          background: rgba(255, 71, 87, 0.08);
          color: var(--danger);
          border: 1px solid rgba(255, 71, 87, 0.2);
          border-radius: 10px;
          padding: 12px 14px;
        }

        .uploadZone {
          background: white;
          border: 3px dashed var(--borderColor);
          border-radius: 20px;
          padding: 60px 40px;
          text-align: center;
          cursor: pointer;
          margin-top: 20px;
        }

        .uploadZone.dragging {
          border-color: var(--accentCyan);
        }

        .uploadZone.uploaded {
          border-color: var(--success);
        }

        .bigSymbol {
          font-size: 48px;
          margin-bottom: 20px;
        }

        .supportedFormats {
          display: inline-block;
          padding: 6px 16px;
          background: #f4f7fa;
          border-radius: 8px;
          font-size: 13px;
          color: var(--textSecondary);
        }

        .analyzingOverlay {
          position: fixed;
          inset: 0;
          background: rgba(0, 31, 63, 0.95);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 999;
        }

        .analyzingContent {
          text-align: center;
          color: white;
          max-width: 560px;
          width: 100%;
          padding: 40px;
        }

        .spinner {
          width: 60px;
          height: 60px;
          border: 4px solid rgba(255,255,255,0.2);
          border-top-color: var(--accentCyan);
          border-radius: 50%;
          margin: 0 auto 24px;
          animation: spin 1s linear infinite;
        }

        .progressBar {
          width: 100%;
          height: 10px;
          background: rgba(255,255,255,0.2);
          border-radius: 999px;
          overflow: hidden;
          margin-top: 18px;
        }

        .progressFill {
          height: 100%;
          background: linear-gradient(90deg, var(--accentCyan), var(--primaryBlue));
          transition: width 0.4s ease;
        }

        .progressText {
          margin-top: 12px;
          font-size: 14px;
          opacity: 0.9;
        }

        .reportsList {
          display: flex;
          flex-direction: column;
          gap: 20px;
        }

        .reportCard, .emptyState {
          background: white;
          border: 1px solid var(--borderColor);
          border-radius: 16px;
          padding: 24px;
        }

        .reportRow {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 16px;
        }

        .reportInfo h3 {
          margin-bottom: 4px;
        }

        .reportDate {
          font-size: 14px;
          color: var(--textSecondary);
        }

        .downloadButton {
          padding: 10px 16px;
          border-radius: 8px;
          background: linear-gradient(135deg, var(--primaryBlue), var(--accentCyan));
          color: white;
          text-decoration: none;
          font-weight: 600;
          white-space: nowrap;
        }

        .downloadButton:hover {
          opacity: 0.92;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        @media (max-width: 768px) {
          .pageHeader h1 {
            font-size: 32px;
          }

          .loginCard {
            padding: 32px 24px;
          }

          .reportRow {
            flex-direction: column;
            align-items: flex-start;
          }
        }
      `}</style>

      {!user ? (
        <LoginPage />
      ) : (
        <>
          <NavigationMenu />
          {currentPage === "reports" ? <ReportsPage /> : <HomePage />}
        </>
      )}
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);