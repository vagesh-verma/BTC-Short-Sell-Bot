import { logger } from './loggerService';

export interface GitHubConfig {
  owner: string;
  repo: string;
  path: string;
  token: string;
}

function buildUrl(owner: string, repo: string, path: string, fileName?: string): string {
  if (!owner || !repo) {
    throw new Error('GitHub Owner and Repo must be configured.');
  }

  const cleanOwner = encodeURIComponent(owner.trim());
  const cleanRepo = encodeURIComponent(repo.trim());
  
  const segments = path.split('/').filter(Boolean);
  if (fileName) {
    segments.push(fileName);
  }
  
  const encodedPath = segments.map(s => {
    return encodeURIComponent(s).replace(/%2B/g, '+');
  }).join('/');
  
  return `https://api.github.com/repos/${cleanOwner}/${cleanRepo}/contents${encodedPath ? '/' + encodedPath : ''}`;
}

const GITHUB_HEADERS = (token: string) => ({
  'Authorization': `token ${token}`,
  'Accept': 'application/vnd.github.v3+json',
  'X-GitHub-Api-Version': '2022-11-28'
});

async function safeFetch(url: string, headers: any, method: string = 'GET', body?: any): Promise<Response> {
  try {
    const options: any = { method, headers };
    if (body) options.body = JSON.stringify(body);
    
    return await fetch(url, options);
  } catch (err: any) {
    if (err.name === 'TypeError' && err.message === 'Failed to fetch') {
      throw new Error('GitHub API unreachable. This could be a network issue or CORS restriction. Please ensure your internet connection is active.');
    }
    throw err;
  }
}

export async function uploadToGitHub(config: GitHubConfig, fileName: string, content: string, message: string) {
  const { owner, repo, path, token } = config;
  const url = buildUrl(owner, repo, path, fileName);

  try {
    // Check if file exists to get its SHA (required for updates)
    let sha: string | undefined;
    const getResponse = await safeFetch(url, GITHUB_HEADERS(token));

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

    const putResponse = await safeFetch(url, {
      ...GITHUB_HEADERS(token),
      'Content-Type': 'application/json'
    }, 'PUT', body);

    if (!putResponse.ok) {
      const errorData = await putResponse.json();
      const errorMsg = errorData.message || 'Failed to upload to GitHub';
      if (putResponse.status === 404) {
        throw new Error(`GitHub Repository not found. Please ensure "https://github.com/${owner}/${repo}" exists and your token has permission to access it. (Error: ${errorMsg})`);
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
    const getResponse = await safeFetch(url, GITHUB_HEADERS(token));

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

    const deleteResponse = await safeFetch(url, {
      ...GITHUB_HEADERS(token),
      'Content-Type': 'application/json'
    }, 'DELETE', { message, sha });

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
    const response = await safeFetch(url, GITHUB_HEADERS(token));

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
    const response = await safeFetch(url, GITHUB_HEADERS(token));

    if (!response.ok) {
      if (response.status === 404) {
        // If it's a 404, we want to know if it's the repo or the path.
        const repoUrl = `https://api.github.com/repos/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}`;
        const repoCheck = await safeFetch(repoUrl, GITHUB_HEADERS(token));
        
        if (!repoCheck.ok) {
          if (repoCheck.status === 404) {
            logger.error(`GitHub Repository "${owner}/${repo}" not found. Please check your repository name and ensure it is public (or your token has access).`);
          } else if (repoCheck.status === 401 || repoCheck.status === 403) {
            logger.error(`GitHub Access Denied. Please check your token permissions for "${owner}/${repo}".`);
          } else {
            logger.error(`GitHub Repo Error: ${repoCheck.statusText} (${repoCheck.status})`);
          }
        } else {
          logger.info(`GitHub path "${path}" not found in repository "${owner}/${repo}". This is normal if you haven't uploaded any models yet.`);
        }
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
