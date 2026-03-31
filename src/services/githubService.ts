import { logger } from './loggerService';

export interface GitHubConfig {
  owner: string;
  repo: string;
  path: string;
  token: string;
}

function buildUrl(owner: string, repo: string, path: string, fileName?: string): string {
  const cleanOwner = encodeURIComponent(owner.trim());
  const cleanRepo = encodeURIComponent(repo.trim());
  
  // Split path into segments and encode each, but preserve '+' as requested by the user.
  // We don't use decodeURIComponent here because it can convert '+' to ' ' in some contexts,
  // and we want to be very explicit about what we encode.
  const segments = path.split('/').filter(Boolean);
  if (fileName) {
    segments.push(fileName);
  }
  
  const encodedPath = segments.map(s => {
    // Encode everything but preserve '+'
    return encodeURIComponent(s).replace(/%2B/g, '+');
  }).join('/');
  
  const url = `https://api.github.com/repos/${cleanOwner}/${cleanRepo}/contents${encodedPath ? '/' + encodedPath : ''}`;
  return url;
}

const GITHUB_HEADERS = (token: string) => ({
  'Authorization': `token ${token}`,
  'Accept': 'application/vnd.github.v3+json',
  'X-GitHub-Api-Version': '2022-11-28' // Standard stable version, or use the one from docs if needed
});

export async function uploadToGitHub(config: GitHubConfig, fileName: string, content: string, message: string) {
  const { owner, repo, path, token } = config;
  const url = buildUrl(owner, repo, path, fileName);

  try {
    // Check if file exists to get its SHA (required for updates)
    let sha: string | undefined;
    const getResponse = await fetch(url, {
      headers: GITHUB_HEADERS(token)
    });

    if (getResponse.ok) {
      const data = await getResponse.json();
      sha = data.sha;
    } else if (getResponse.status === 404) {
      // 404 on GET is fine, it just means the file doesn't exist yet.
      // However, if the repo doesn't exist, the subsequent PUT will fail with 404 too.
    }

    const body = {
      message,
      content: btoa(unescape(encodeURIComponent(content))), // Robust Base64 encode for UTF-8
      sha
    };

    const putResponse = await fetch(url, {
      method: 'PUT',
      headers: {
        ...GITHUB_HEADERS(token),
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(body)
    });

    if (!putResponse.ok) {
      const errorData = await putResponse.json();
      const errorMsg = errorData.message || 'Failed to upload to GitHub';
      if (putResponse.status === 404) {
        throw new Error(`GitHub Repository or Path not found. Please check your Owner ("${owner}") and Repo ("${repo}") settings. (Error: ${errorMsg})`);
      }
      throw new Error(errorMsg);
    }

    logger.success(`Successfully uploaded ${fileName} to GitHub.`);
  } catch (err) {
    logger.error(`GitHub upload error: ${err}`);
    throw err;
  }
}

export async function deleteFromGitHub(config: GitHubConfig, fileName: string, message: string) {
  const { owner, repo, path, token } = config;
  const url = buildUrl(owner, repo, path, fileName);

  try {
    // Must get SHA first
    const getResponse = await fetch(url, {
      headers: GITHUB_HEADERS(token)
    });

    if (!getResponse.ok) {
      if (getResponse.status === 404) {
        logger.info(`File ${fileName} not found on GitHub, skipping deletion.`);
        return;
      }
      const errorData = await getResponse.json();
      throw new Error(errorData.message || 'Failed to find file on GitHub');
    }

    const data = await getResponse.json();
    const sha = data.sha;

    const deleteResponse = await fetch(url, {
      method: 'DELETE',
      headers: {
        ...GITHUB_HEADERS(token),
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        message,
        sha
      })
    });

    if (!deleteResponse.ok) {
      const errorData = await deleteResponse.json();
      throw new Error(errorData.message || 'Failed to delete from GitHub');
    }

    logger.success(`Successfully deleted ${fileName} from GitHub.`);
  } catch (err) {
    logger.error(`GitHub deletion error: ${err}`);
    throw err;
  }
}

export async function fetchFromGitHub(config: GitHubConfig, fileName: string): Promise<string> {
  const { owner, repo, path, token } = config;
  const url = buildUrl(owner, repo, path, fileName);
  logger.info(`Fetching from GitHub: ${url}`);

  try {
    const response = await fetch(url, {
      headers: GITHUB_HEADERS(token)
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMsg = errorData.message || `Failed to fetch ${fileName}`;
      if (response.status === 404) {
        throw new Error(`File "${fileName}" not found in GitHub path "${path}". (Error: ${errorMsg})`);
      }
      throw new Error(errorMsg);
    }

    const data = await response.json();
    // GitHub returns content as base64
    return decodeURIComponent(escape(atob(data.content)));
  } catch (err) {
    logger.error(`GitHub fetch error: ${err}`);
    throw err;
  }
}

export async function listFromGitHub(config: GitHubConfig): Promise<any[]> {
  const { owner, repo, path, token } = config;
  const url = buildUrl(owner, repo, path);

  try {
    const response = await fetch(url, {
      headers: GITHUB_HEADERS(token)
    });

    if (!response.ok) {
      if (response.status === 404) {
        // If it's a 404, it could be that the path doesn't exist yet, OR the repo is wrong.
        // We'll return an empty array but log a warning if it might be a repo issue.
        logger.info(`GitHub path "${path}" not found or repository "${owner}/${repo}" is inaccessible.`);
        return [];
      }
      const errorData = await response.json();
      throw new Error(errorData.message || `Failed to list ${path} from GitHub`);
    }

    return await response.json();
  } catch (err) {
    logger.error(`GitHub list error: ${err}`);
    throw err;
  }
}
